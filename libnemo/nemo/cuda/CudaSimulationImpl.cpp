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
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>

#include <nemo/exception.hpp>
#include <nemo/NetworkImpl.hpp>
#include <nemo/fixedpoint.hpp>

#include "CycleCounters.hpp"
#include "DeviceAssertions.hpp"
#include "ThalamicInput.hpp"
#include "bitvector.hpp"
#include "exception.hpp"

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
	m_neurons(net, m_mapper),
	m_cm(net, conf, m_mapper),
	m_recentFiring(m_mapper.partitionCount(), conf.cudaPartitionSize(), false, 2),
	//! \todo seed properly from configuration
	m_thalamicInput(net, m_mapper),
	//! \todo use pinned host memory here
	m_firingStimulus(m_mapper.partitionCount(), BV_WORD_PITCH, false),
	//! \todo allow external users to directly use the host buffer
	m_currentStimulus(m_mapper.partitionCount(), conf.cudaPartitionSize(), true),
	m_firingOutput(m_mapper, conf.cudaFiringBufferLength()),
	m_cycleCounters(m_mapper.partitionCount(), usingStdp()),
	m_deviceAssertions(m_mapper.partitionCount()),
	m_pitch32(0),
	m_pitch64(0),
	md_fstim(NULL),
	md_istim(NULL)
{
	configureStdp(conf.stdpFunction());
	setPitch();
	//! \todo do this configuration as part of CM setup
	CUDA_SAFE_CALL(configureKernel(m_cm.maxDelay(), m_pitch32, m_pitch64));
}



SimulationImpl::~SimulationImpl()
{
	finishSimulation();
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



void
SimulationImpl::setFiringStimulus(const std::vector<unsigned>& nidx)
{
	if(nidx.empty()) {
		md_fstim = NULL;
		return;
	}

	//! \todo use internal host buffer with pinned memory instead
	size_t pitch = m_firingStimulus.wordPitch();
	std::vector<uint32_t> hostArray(m_firingStimulus.size(), 0);

	for(std::vector<unsigned>::const_iterator i = nidx.begin();
			i != nidx.end(); ++i) {
		//! \todo should check that this neuron exists
		DeviceIdx dev = m_mapper.deviceIdx(*i);
		size_t word = dev.partition * pitch + dev.neuron / 32;
		size_t bit = dev.neuron % 32;
		hostArray[word] |= 1 << bit;
	}

	memcpyToDevice(m_firingStimulus.deviceData(), hostArray, m_mapper.partitionCount() * pitch);
	md_fstim = m_firingStimulus.deviceData();
}



void
SimulationImpl::clearFiringStimulus()
{
	md_fstim = NULL;
}



void
SimulationImpl::setCurrentStimulus(const std::vector<fix_t>& current)
{
	if(current.empty()) {
		md_istim = NULL;
		return;
	}

	m_currentStimulus.fill(0);

	/* The indices into 'current' are 0-based local indices. We need to
	 * translate this into the appropriately mapped device indices. In
	 * practice, the mapping currently is such that we should be able to copy
	 * all the weights for a single partition (or even whole network) at the
	 * same time, rather than having to do this on a per-neuron basis. If the
	 * current copying scheme turns out to be a bottleneck, modify this. */
	//! \todo in public API get vector of neuron/current pairs instead.
	nidx_t neuron = m_mapper.minHostIdx();
	for(std::vector<fix_t>::const_iterator i = current.begin();
			i != current.end(); ++i, ++neuron) {
		DeviceIdx dev = m_mapper.deviceIdx(neuron);
		m_currentStimulus.setNeuron(dev.partition, dev.neuron, *i);
	}
	m_currentStimulus.copyToDevice();
	md_istim = m_currentStimulus.deviceData();
}



void
SimulationImpl::clearCurrentStimulus()
{
	md_istim = NULL;
}



void
checkPitch(size_t expected, size_t found)
{
	if(expected != found) {
		std::ostringstream msg;
		msg << "Simulation::checkPitch: pitch mismatch in device memory allocation. "
			"Found " << found << ", expected " << expected << std::endl;
		throw nemo::exception(NEMO_CUDA_MEMORY_ERROR, msg.str());
	}
}


size_t
SimulationImpl::d_allocated() const
{
	return m_firingStimulus.d_allocated()
		+ m_currentStimulus.d_allocated()
		+ m_recentFiring.d_allocated()
		+ m_neurons.d_allocated()
		+ m_firingOutput.d_allocated()
		+ m_thalamicInput.d_allocated()
		+ m_cm.d_allocated();
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
SimulationImpl::setPitch()
{
	size_t pitch1 = m_firingStimulus.wordPitch();
	m_pitch32 = m_neurons.wordPitch();
	m_pitch64 = m_recentFiring.wordPitch();
	//! \todo fold thalamic input into neuron parameters
	checkPitch(m_pitch32, m_thalamicInput.wordPitch());
	checkPitch(m_pitch32, m_currentStimulus.wordPitch());
	checkPitch(pitch1, m_firingOutput.wordPitch());
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
SimulationImpl::step()
{
	m_timer.step();
	uint32_t* d_fout = m_firingOutput.step();
	::stepSimulation(
			m_mapper.partitionCount(),
			usingStdp(),
			m_timer.elapsedSimulation(),
			m_recentFiring.deviceData(),
			m_neurons.deviceData(),
			m_thalamicInput.deviceRngState(),
			m_thalamicInput.deviceSigma(),
			md_fstim,
			md_istim,
			d_fout,
			m_cm.d_fcm(),
			m_cm.outgoingCount(),
			m_cm.outgoing(),
			m_cm.incomingHeads(),
			m_cm.incoming(),
			m_cycleCounters.data(),
			m_cycleCounters.pitch());

	/* Must clear stimulus pointers in case the low-level interface is used and
	 * the user does not provide any fresh stimulus */
	clearFiringStimulus();
	clearCurrentStimulus();

	cudaError_t status = cudaGetLastError();
	if(status != cudaSuccess) {
		//! \todo add cycle number?
		throw nemo::exception(NEMO_CUDA_INVOCATION_ERROR, cudaGetErrorString(status));
	}

	m_deviceAssertions.check(m_timer.elapsedSimulation());
}



void
SimulationImpl::applyStdp(float reward)
{
	if(!usingStdp()) {
		throw exception(NEMO_LOGIC_ERROR, "applyStdp called when STDP not in use");
		return;
	}

	if(reward == 0.0f) {
		m_cm.clearStdpAccumulator();
	} else  {
		::applyStdp(
				m_cycleCounters.dataApplySTDP(),
				m_cycleCounters.pitchApplySTDP(),
				m_mapper.partitionCount(),
				m_cm.fractionalBits(),
				m_cm.d_fcm(),
				m_stdpFn.maxWeight(),
				m_stdpFn.minWeight(),
				reward);
	}

	m_deviceAssertions.check(m_timer.elapsedSimulation());
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
	return m_firingOutput.readFiring(cycles, nidx);
}


void
SimulationImpl::flushFiringBuffer()
{
	m_firingOutput.flushBuffer();
}


void
SimulationImpl::finishSimulation()
{
	//! \todo perhaps clear device data here instead of in dtor
	if(m_conf.loggingEnabled()) {
		m_cycleCounters.printCounters(std::cout);
		//! \todo add time summary
	}
}


unsigned long
SimulationImpl::elapsedWallclock() const
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	return m_timer.elapsedWallclock();
}


unsigned long
SimulationImpl::elapsedSimulation() const
{
	return m_timer.elapsedSimulation();
}



void
SimulationImpl::resetTimer()
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	m_timer.reset();
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


unsigned
SimulationImpl::getFractionalBits() const
{
	return m_cm.fractionalBits();
}

	} // end namespace cuda
} // end namespace nemo
