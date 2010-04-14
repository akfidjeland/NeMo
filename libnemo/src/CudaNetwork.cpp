/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "CudaNetwork.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>

#include "DeviceAssertions.hpp"
#include "FiringOutput.hpp"
#include "ConnectivityMatrix.hpp"
#include "CycleCounters.hpp"
#include "ThalamicInput.hpp"
#include "util.h"
#include "log.hpp"
#include "fixedpoint.hpp"
#include "bitvector.hpp"
#include "except.hpp"

#include "device_assert.cu_h"
#include "kernel.cu_h"
#include "kernel.hpp"


namespace nemo {
	namespace cuda {


CudaNetwork::CudaNetwork(
		const nemo::Network& net,
		const nemo::Configuration& conf) :
	m_conf(conf),
	m_partitionCount(0),
	//! \todo get rid of member variable
	m_maxPartitionSize(conf.cudaMaxPartitionSize()),
	m_neurons(net, conf.cudaMaxPartitionSize()),
	m_cm(new ConnectivityMatrix(net, conf.cudaMaxPartitionSize(), conf.loggingEnabled())),
	m_recentFiring(NULL),
	m_thalamicInput(NULL),
	m_firingStimulus(NULL),
	m_firingOutput(NULL),
	m_cycleCounters(NULL),
	m_deviceAssertions(NULL),
	m_pitch32(0),
	m_pitch64(0)
{
	if(conf.stdpFunction() != NULL) {
		m_stdpFn = *conf.stdpFunction();
	}

	configureStdp();

	//! \todo merge with init list
	//! \todo remove m_partitionCount member variable
	m_partitionCount = m_neurons.partitionCount();
	m_deviceAssertions = new DeviceAssertions(m_partitionCount);
	m_firingOutput = new FiringOutput(m_partitionCount, m_maxPartitionSize, conf.cudaFiringBufferLength());
	m_recentFiring = new NVector<uint64_t>(m_partitionCount, m_maxPartitionSize, false, 2);
	//! \todo seed properly from outside function
	m_thalamicInput = new ThalamicInput(m_partitionCount, m_maxPartitionSize, 0);
	//! \todo change NeuronParameters API for this function
	m_neurons.setSigma(*m_thalamicInput);
	m_thalamicInput->moveToDevice();
	m_cycleCounters = new CycleCounters(m_partitionCount, usingStdp());
	m_firingStimulus = new NVector<uint32_t>(m_partitionCount, BV_WORD_PITCH, false);

	setPitch();
	//! \todo do this configuration as part of CM setup
	configureKernel(m_cm->maxDelay(), m_pitch32, m_pitch64);
	resetTimer();
}


CudaNetwork::~CudaNetwork()
{
	finishSimulation();
	//! \todo used shared_ptr instead to deal with this
	if(m_deviceAssertions) delete m_deviceAssertions;
	if(m_firingOutput) delete m_firingOutput;
	if(m_recentFiring) delete m_recentFiring;
	if(m_firingStimulus) delete m_firingStimulus;
	if(m_thalamicInput) delete m_thalamicInput;
	if(m_cycleCounters) delete m_cycleCounters;
	if(m_cm) delete m_cm;
}


int CudaNetwork::s_device = -1;

int
CudaNetwork::selectDevice()
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
	cudaGetDevice(&dev);
	CUDA_SAFE_CALL(cudaChooseDevice(&dev, &prop));

	// 9999.9999 is the 'emulation device' which is always present
	if(prop.major == 9999 || prop.minor == 9999) {
		std::cerr << "No physical devices available" << std::endl;
		return -1;
	}

	// 1.2 requires for shared memory atomics
	if(prop.major <= 1 && prop.minor < 2) {
		std::cerr << "No device with compute capability 1.2 available" << std::endl;
		return -1;
	}

	CUDA_SAFE_CALL(cudaSetDevice(dev));
	s_device = dev;
	return dev;
}



//! \todo simplify
void
CudaNetwork::configureStdp()
{
	if(!usingStdp()) {
		return;
	}

	const std::vector<float>& flfn = m_stdpFn.function();
	std::vector<fix_t> fxfn(flfn.size());
	uint fb = m_cm->fractionalBits();
	for(uint i=0; i < fxfn.size(); ++i) {
		fxfn.at(i) = fx_toFix(flfn[i], fb);
	}
	::configureStdp(m_stdpFn.preFireWindow(),
			m_stdpFn.postFireWindow(),
			m_stdpFn.potentiationBits(),
			m_stdpFn.depressionBits(),
			const_cast<fix_t*>(&fxfn[0]));
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
CudaNetwork::setFiringStimulus(const std::vector<uint>& nidx)
{
	if(nidx.empty())
		return NULL;

	//! \todo use internal host buffer with pinned memory instead
	size_t pitch = m_firingStimulus->wordPitch();
	std::vector<uint32_t> hostArray(m_firingStimulus->size(), 0);

	for(std::vector<uint>::const_iterator i = nidx.begin();
			i != nidx.end(); ++i) {
		//! \todo share this translation with NeuronParameters and CMImpl
		size_t nn = *i % m_maxPartitionSize;
		size_t pn = *i / m_maxPartitionSize;
		//! \todo should check the size of this particular partition
		assert(nn < m_maxPartitionSize );
		assert(pn < m_partitionCount);
		size_t word = pn * pitch + nn / 32;
		size_t bit = nn % 32;
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
		ERROR("CudaNetwork::checkPitch: pitch mismatch in device memory allocation. Found %d, expected %d\n",
				(int) found, (int) expected);
	}
}


size_t
CudaNetwork::d_allocated() const
{
	size_t total = 0;
	total += m_firingStimulus ? m_firingStimulus->d_allocated()   : 0;
	total += m_recentFiring   ? m_recentFiring->d_allocated()     : 0;
	total += m_neurons.d_allocated();
	total += m_firingOutput   ? m_firingOutput->d_allocated()     : 0;
	total += m_thalamicInput  ? m_thalamicInput->d_allocated()    : 0;
	total += m_cm             ? m_cm->d_allocated()             : 0;
	return total;
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
CudaNetwork::setPitch()
{
	size_t pitch1 = m_firingStimulus->wordPitch();
	m_pitch32 = m_neurons.wordPitch();
	m_pitch64 = m_recentFiring->wordPitch();
	//! \todo fold thalamic input into neuron parameters
	checkPitch(m_pitch32, m_thalamicInput->wordPitch());
	checkPitch(pitch1, m_firingOutput->wordPitch());
	bv_setPitch(pitch1);
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------



unsigned long
CudaNetwork::elapsedWallclock() const
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	return m_timer.elapsedWallclock();
}



unsigned long
CudaNetwork::elapsedSimulation() const
{
	return m_timer.elapsedSimulation();
}



void
CudaNetwork::resetTimer()
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	m_timer.reset();
}




//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


bool
CudaNetwork::usingStdp() const
{
	return m_stdpFn.enabled();
}




void
CudaNetwork::stepSimulation(const std::vector<uint>& fstim)
{
	/* A 32-bit counter can count up to around 4M seconds which is around 1200
	 * hours or 50 days */
	//! \todo use a 64-bit counter instead
	if(m_cycle == ~0U) {
		throw std::overflow_error("Cycle counter overflow");
	}
	m_cycle += 1;
	m_timer.step();

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
			m_cm->d_fcm(),
			m_cm->outgoingCount(),
			m_cm->outgoing(),
			m_cm->incomingHeads(),
			m_cm->incoming(),
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
CudaNetwork::applyStdp(float reward)
{
	if(!usingStdp()) {
		//! \todo issue a warning here?
		return;
	}

	if(reward == 0.0f) {
		m_cm->clearStdpAccumulator();
	} else  {
		::applyStdp(
				m_cycleCounters->dataApplySTDP(),
				m_cycleCounters->pitchApplySTDP(),
				m_partitionCount,
				m_cm->fractionalBits(),
				m_cm->d_fcm(),
				m_stdpFn.maxWeight(),
				m_stdpFn.minWeight(),
				reward);
	}

	m_deviceAssertions->check(m_cycle);
}



void
CudaNetwork::getSynapses(unsigned sn,
		const std::vector<unsigned>** tn,
		const std::vector<unsigned>** d,
		const std::vector<float>** w,
		const std::vector<unsigned char>** p)
{
	return m_cm->getSynapses(sn, tn, d, w, p);
}



uint
CudaNetwork::readFiring(
		const std::vector<uint>** cycles,
		const std::vector<uint>** nidx)
{
	return m_firingOutput->readFiring(cycles, nidx);
}


void
CudaNetwork::flushFiringBuffer()
{
	m_firingOutput->flushBuffer();
}


void
CudaNetwork::finishSimulation()
{
	//! \todo perhaps clear device data here instead of in dtor
	if(m_conf.loggingEnabled()) {
		m_cycleCounters->printCounters(std::cout);
		//! \todo add time summary
	}
}



	} // end namespace cuda
} // end namespace nemo
