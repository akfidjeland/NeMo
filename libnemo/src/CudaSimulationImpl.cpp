/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "CudaSimulationImpl.hpp"

#include <vector>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>

#include "CycleCounters.hpp"
#include "DeviceAssertions.hpp"
#include "FiringOutput.hpp"
#include "NetworkImpl.hpp"
#include "ThalamicInput.hpp"
#include "fixedpoint.hpp"
#include "bitvector.hpp"
#include "except.hpp"

#include "device_assert.cu_h"
#include "kernel.cu_h"
#include "kernel.hpp"



namespace nemo {
	namespace cuda {


SimulationImpl::SimulationImpl(
		const nemo::NetworkImpl& net,
		const nemo::ConfigurationImpl& conf) :
	m_mapper(net, conf.cudaPartitionSize()),
	m_conf(conf),
	m_partitionCount(0),
	//! \todo get rid of member variable
	m_maxPartitionSize(conf.cudaPartitionSize()),
	m_neurons(net, m_mapper),
	m_cycle(0),
	m_cm(net, m_mapper, conf.cudaPartitionSize(), conf.loggingEnabled()),
	m_recentFiring(NULL),
	m_thalamicInput(NULL),
	m_firingStimulus(NULL),
	m_firingOutput(NULL),
	m_cycleCounters(NULL),
	m_deviceAssertions(NULL),
	m_pitch32(0),
	m_pitch64(0)
{
	configureStdp(conf.stdpFunction());

	//! \todo merge with init list
	//! \todo remove m_partitionCount member variable
	m_partitionCount = m_mapper.partitionCount();
	m_deviceAssertions = new DeviceAssertions(m_partitionCount);
	m_firingOutput = new FiringOutput(m_partitionCount, m_maxPartitionSize, conf.cudaFiringBufferLength());
	m_recentFiring = new NVector<uint64_t>(m_partitionCount, m_maxPartitionSize, false, 2);
	//! \todo seed properly from configuration
	m_thalamicInput = new ThalamicInput(net, m_mapper, m_partitionCount, m_maxPartitionSize);
	m_cycleCounters = new CycleCounters(m_partitionCount, usingStdp());
	m_firingStimulus = new NVector<uint32_t>(m_partitionCount, BV_WORD_PITCH, false);

	setPitch();
	//! \todo do this configuration as part of CM setup
	CUDA_SAFE_CALL(configureKernel(m_cm.maxDelay(), m_pitch32, m_pitch64));
}


SimulationImpl::~SimulationImpl()
{
	finishSimulation();
	//! \todo used shared_ptr instead to deal with this
	if(m_deviceAssertions) delete m_deviceAssertions;
	if(m_firingOutput) delete m_firingOutput;
	if(m_recentFiring) delete m_recentFiring;
	if(m_firingStimulus) delete m_firingStimulus;
	if(m_thalamicInput) delete m_thalamicInput;
	if(m_cycleCounters) delete m_cycleCounters;
}


#ifdef __DEVICE_EMULATION__
int SimulationImpl::s_device = 0;
#else
int SimulationImpl::s_device = -1;
#endif

int
SimulationImpl::selectDevice()
{
	/*! \todo might want to use thread-local, rather than process-local storage
	 * for s_device in order to support multiple threads */
	if(s_device != -1) {
		return s_device;
	}

	int dev;
	cudaDeviceProp prop;
	prop.major = 1;
	prop.minor = 2;

	CUDA_SAFE_CALL(cudaChooseDevice(&dev, &prop));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));

	/* 9999.9999 is the 'emulation device' which is always present. Unless the
	 * library was built specifically for emulation mode, this should be
	 * considered an error. */
	if(prop.major == 9999 || prop.minor == 9999) {
		//! \todo perhaps throw exception instead?
		std::cerr << "No physical devices available" << std::endl;
		return -1;
	}

	// 1.2 required for shared memory atomics
	if(prop.major <= 1 && prop.minor < 2) {
		std::cerr << "No device with compute capability 1.2 available" << std::endl;
		return -1;
	}

	CUDA_SAFE_CALL(cudaSetDevice(dev));
	s_device = dev;
	return dev;
}



int
SimulationImpl::setDevice(int dev)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, dev);

	//! \todo throw exceptions instead here?
	// 9999.9999 is the 'emulation device' which is always present
	if(prop.major == 9999 || prop.minor == 9999) {
		std::cerr << "No physical devices available" << std::endl;
		return -1;
	}

	// 1.2 required for shared memory atomics
	if(prop.major <= 1 && prop.minor < 2) {
		std::cerr << "Device has compute capability less than 1.2" << std::endl;
		return -1;
	}

	CUDA_SAFE_CALL(cudaSetDevice(dev));
	s_device = dev;
	return dev;

}



void
SimulationImpl::configureStdp(const STDP<float>& stdp)
{
	if(!stdp.enabled()) {
		return;
	}

	m_stdpFn = stdp;

	const std::vector<float>& flfn = m_stdpFn.function();
	std::vector<fix_t> fxfn(flfn.size());
	unsigned fb = m_cm.fractionalBits();
	for(unsigned i=0; i < fxfn.size(); ++i) {
		fxfn.at(i) = fx_toFix(flfn[i], fb);
	}
	CUDA_SAFE_CALL(
		::configureStdp(m_stdpFn.preFireWindow(),
			m_stdpFn.postFireWindow(),
			m_stdpFn.potentiationBits(),
			m_stdpFn.depressionBits(),
			const_cast<fix_t*>(&fxfn[0])));
}



/*! Copy firing stimulus from host to device. Array indices only tested in
 * debugging mode.
 * 
 * \param count
 *		Number of neurons whose firing should be forced
 * \param nidx
 * 		Neuron indices of neurons whose firing should be forced
 *
 * \return 
 *		Pointer to pass to kernel (which is NULL if there's no firing data).
 */
uint32_t*
SimulationImpl::setFiringStimulus(const std::vector<unsigned>& nidx)
{
	if(nidx.empty())
		return NULL;

	//! \todo use internal host buffer with pinned memory instead
	size_t pitch = m_firingStimulus->wordPitch();
	std::vector<uint32_t> hostArray(m_firingStimulus->size(), 0);

	for(std::vector<unsigned>::const_iterator i = nidx.begin();
			i != nidx.end(); ++i) {
		//! \todo should check that this neuron exists
		DeviceIdx dev = m_mapper.deviceIdx(*i);
		size_t word = dev.partition * pitch + dev.neuron / 32;
		size_t bit = dev.neuron % 32;
		hostArray[word] |= 1 << bit;
	}

	CUDA_SAFE_CALL(cudaMemcpy(
				m_firingStimulus->deviceData(),
				&hostArray[0],
				m_partitionCount * m_firingStimulus->bytePitch(),
				cudaMemcpyHostToDevice));

	return m_firingStimulus->deviceData();
}



void
checkPitch(size_t expected, size_t found)
{
	if(expected != found) {
		std::ostringstream msg;
		msg << "Simulation::checkPitch: pitch mismatch in device memory allocation. "
			"Found " << found << ", expected " << expected << std::endl;
		throw std::runtime_error(msg.str());
	}
}


size_t
SimulationImpl::d_allocated() const
{
	size_t total = 0;
	total += m_firingStimulus ? m_firingStimulus->d_allocated()   : 0;
	total += m_recentFiring   ? m_recentFiring->d_allocated()     : 0;
	total += m_neurons.d_allocated();
	total += m_firingOutput   ? m_firingOutput->d_allocated()     : 0;
	total += m_thalamicInput  ? m_thalamicInput->d_allocated()    : 0;
	total += m_cm.d_allocated();
	return total;
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
SimulationImpl::setPitch()
{
	size_t pitch1 = m_firingStimulus->wordPitch();
	m_pitch32 = m_neurons.wordPitch();
	m_pitch64 = m_recentFiring->wordPitch();
	//! \todo fold thalamic input into neuron parameters
	checkPitch(m_pitch32, m_thalamicInput->wordPitch());
	checkPitch(pitch1, m_firingOutput->wordPitch());
	CUDA_SAFE_CALL(bv_setPitch(pitch1));
}



//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


bool
SimulationImpl::usingStdp() const
{
	return m_stdpFn.enabled();
}




void
SimulationImpl::step(const std::vector<unsigned>& fstim)
{
	/* A 32-bit counter can count up to around 4M seconds which is around 1200
	 * hours or 50 days */
	//! \todo use a 64-bit counter instead
	if(m_cycle == ~0U) {
		throw std::overflow_error("Cycle counter overflow");
	}
	m_cycle += 1;
#ifdef INCLUDE_TIMING_API
	m_timer.step();
#endif

	uint32_t* d_fstim = setFiringStimulus(fstim);
	uint32_t* d_fout = m_firingOutput->step();
	::stepSimulation(
			m_partitionCount,
			usingStdp(),
			m_cycle,
			m_recentFiring->deviceData(),
			m_neurons.deviceData(),
			m_thalamicInput->deviceRngState(),
			m_thalamicInput->deviceSigma(),
			d_fstim, 
			d_fout,
			m_cm.d_fcm(),
			m_cm.outgoingCount(),
			m_cm.outgoing(),
			m_cm.incomingHeads(),
			m_cm.incoming(),
			m_cycleCounters->data(),
			m_cycleCounters->pitch());

	cudaError_t status = cudaGetLastError();
	if(status != cudaSuccess) {
		//! \todo add cycle number?
		throw KernelInvocationError(status);
	}

	m_deviceAssertions->check(m_cycle);
}


void
SimulationImpl::applyStdp(float reward)
{
	if(!usingStdp()) {
		//! \todo issue a warning here?
		return;
	}

	if(reward == 0.0f) {
		m_cm.clearStdpAccumulator();
	} else  {
		::applyStdp(
				m_cycleCounters->dataApplySTDP(),
				m_cycleCounters->pitchApplySTDP(),
				m_partitionCount,
				m_cm.fractionalBits(),
				m_cm.d_fcm(),
				m_stdpFn.maxWeight(),
				m_stdpFn.minWeight(),
				reward);
	}

	m_deviceAssertions->check(m_cycle);
}



void
SimulationImpl::getSynapses(unsigned sn,
		const std::vector<unsigned>** tn,
		const std::vector<unsigned>** d,
		const std::vector<float>** w,
		const std::vector<unsigned char>** p)
{
	return m_cm.getSynapses(sn, tn, d, w, p);
}



unsigned
SimulationImpl::readFiring(
		const std::vector<unsigned>** cycles,
		const std::vector<unsigned>** nidx)
{
	return m_firingOutput->readFiring(cycles, nidx);
}


void
SimulationImpl::flushFiringBuffer()
{
	m_firingOutput->flushBuffer();
}


void
SimulationImpl::finishSimulation()
{
	//! \todo perhaps clear device data here instead of in dtor
	if(m_conf.loggingEnabled()) {
		m_cycleCounters->printCounters(std::cout);
		//! \todo add time summary
	}
}


unsigned long
SimulationImpl::elapsedWallclock() const
{
#ifdef INCLUDE_TIMING_API
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	return m_timer.elapsedWallclock();
#else
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"elapsedWallclock is not supported in this version");
#endif
}


unsigned long
SimulationImpl::elapsedSimulation() const
{
#ifdef INCLUDE_TIMING_API
	return m_timer.elapsedSimulation();
#else
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"elapsedSimulation is not supported in this version");
#endif
}



void
SimulationImpl::resetTimer()
{
#ifdef INCLUDE_TIMING_API
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	m_timer.reset();
#endif
}



unsigned
SimulationImpl::defaultPartitionSize()
{
	return MAX_PARTITION_SIZE;
}



unsigned
SimulationImpl::defaultFiringBufferLength()
{
	return FiringOutput::defaultBufferLength();
}

	} // end namespace cuda
} // end namespace nemo
