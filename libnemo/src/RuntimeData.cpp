#include "RuntimeData.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>

#include "DeviceAssertions.hpp"
#include "FiringOutput.hpp"
#include "ConnectivityMatrix.hpp"
#include "CycleCounters.hpp"
#include "NeuronParameters.hpp"
#include "ThalamicInput.hpp"
#include "util.h"
#include "log.hpp"
#include "fixedpoint.hpp"
#include "bitvector.hpp"
#include "except.hpp"

#include "partitionConfiguration.cu_h"
#include "kernel.hpp"
#include "kernel.cu_h"
#include "device_assert.cu_h"



namespace nemo {

RuntimeData::RuntimeData(bool setReverse, unsigned int maxReadPeriod) :
	m_partitionCount(0),
	m_maxPartitionSize(MAX_PARTITION_SIZE),
	m_neurons(new NeuronParameters(m_maxPartitionSize)),
	m_cm(new ConnectivityMatrix(m_maxPartitionSize, setReverse)),
	m_recentFiring(NULL),
	m_thalamicInput(NULL),
	m_firingStimulus(NULL),
	m_firingOutput(NULL),
	m_cycleCounters(NULL),
	m_deviceAssertions(NULL),
	m_pitch32(0),
	m_pitch64(0),
	m_deviceDirty(true),
	m_maxReadPeriod(maxReadPeriod)
{
	//! \todo use this in move to device
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&m_deviceProperties, device);
}



RuntimeData::RuntimeData(bool setReverse,
		unsigned maxReadPeriod,
		unsigned maxPartitionSize) :
	m_partitionCount(0),
	m_maxPartitionSize(maxPartitionSize),
	m_neurons(new NeuronParameters(m_maxPartitionSize)),
	m_cm(new ConnectivityMatrix(m_maxPartitionSize, setReverse)),
	m_recentFiring(NULL),
	m_thalamicInput(NULL),
	m_firingStimulus(NULL),
	m_firingOutput(NULL),
	m_cycleCounters(NULL),
	m_deviceAssertions(NULL),
	m_pitch32(0),
	m_pitch64(0),
	m_deviceDirty(true),
	m_maxReadPeriod(maxReadPeriod)
{
	//! \todo use this in move to device
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&m_deviceProperties, device);
}


RuntimeData::~RuntimeData()
{
	//! \todo used shared_ptr instead to deal with this
	if(m_deviceAssertions) delete m_deviceAssertions;
	if(m_firingOutput) delete m_firingOutput;
	if(m_recentFiring) delete m_recentFiring;
	if(m_firingStimulus) delete m_firingStimulus;
	if(m_thalamicInput) delete m_thalamicInput;
	if(m_cycleCounters) delete m_cycleCounters;
	delete m_cm;
	delete m_neurons;
}



void
RuntimeData::configureStdp()
{
	if(!stdpFn.enabled()) {
		return;
	}

	const std::vector<float>& flfn = stdpFn.function();
	std::vector<fix_t> fxfn(flfn.size());
	uint fb = m_cm->fractionalBits();
	for(uint i=0; i < fxfn.size(); ++i) {
		fxfn.at(i) = fixedPoint(flfn[i], fb);
	}
	::configureStdp(stdpFn.preFireWindow(),
			stdpFn.postFireWindow(),
			stdpFn.potentiationBits(),
			stdpFn.depressionBits(),
			const_cast<fix_t*>(&fxfn[0]));
}



void
RuntimeData::moveToDevice()
{
	if(m_deviceDirty) {
		m_cm->moveToDevice();
		m_neurons->moveToDevice();
		configureStdp();
		m_partitionCount = m_neurons->partitionCount();
		m_deviceAssertions = new DeviceAssertions(m_partitionCount);
		m_firingOutput = new FiringOutput(m_partitionCount, m_maxPartitionSize, m_maxReadPeriod);
		m_recentFiring = new NVector<uint64_t>(m_partitionCount, m_maxPartitionSize, false, 2);
		//! \todo seed properly from outside function
		m_thalamicInput = new ThalamicInput(m_partitionCount, m_maxPartitionSize, 0);
		m_neurons->setSigma(*m_thalamicInput);
		m_thalamicInput->moveToDevice();
		m_cycleCounters = new CycleCounters(m_partitionCount, m_deviceProperties.clockRate);
		m_firingStimulus = new NVector<uint32_t>(m_partitionCount, BV_WORD_PITCH, false);

		setPitch();
	    m_deviceDirty = false;
	}
}



bool
RuntimeData::deviceDirty() const
{
	return m_deviceDirty;
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
RuntimeData::setFiringStimulus(const std::vector<uint>& nidx)
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
		ERROR("RuntimeData::checkPitch: pitch mismatch in device memory allocation. Found %d, expected %d\n",
				(int) found, (int) expected);
	}
}


size_t
RuntimeData::d_allocated() const
{
	size_t total = 0;
	total += m_firingStimulus ? m_firingStimulus->d_allocated()   : 0;
	total += m_recentFiring   ? m_recentFiring->d_allocated()     : 0;
	total += m_neurons        ? m_neurons->d_allocated()        : 0;
	total += m_firingOutput   ? m_firingOutput->d_allocated()     : 0;
	total += m_thalamicInput  ? m_thalamicInput->d_allocated()    : 0;
	total += m_cm             ? m_cm->d_allocated()             : 0;
	return total;
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
RuntimeData::setPitch()
{
	size_t pitch1 = m_firingStimulus->wordPitch();
	m_pitch32 = m_neurons->wordPitch();
	m_pitch64 = m_recentFiring->wordPitch();
	//! \todo fold thalamic input into neuron parameters
	checkPitch(m_pitch32, m_thalamicInput->wordPitch());
	checkPitch(pitch1, m_firingOutput->wordPitch());
	bv_setPitch(pitch1);
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


long
RuntimeData::elapsed()
{
    syncSimulation();
	return m_timer.elapsed();
}



void
RuntimeData::setStart()
{
	m_timer.reset();
}




//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


bool
RuntimeData::usingStdp() const
{
	return stdpFn.enabled();
}



void
RuntimeData::addNeuron(
		unsigned int idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	m_neurons->addNeuron(idx, a, b, c, d, u, v, sigma);
}



void
RuntimeData::addSynapses(
		uint source,
		const std::vector<uint>& targets,
		const std::vector<uint>& delays,
		const std::vector<float>& weights,
		const std::vector<unsigned char> is_plastic)
{
	m_cm->addSynapses(source, targets, delays, weights, is_plastic);
}



void
RuntimeData::syncSimulation()
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
}



void
RuntimeData::startSimulation()
{
	//! \todo merge this function with moveToDevice
	if(deviceDirty()) {
		moveToDevice();
		//! \todo do this configuration as part of CM setup
		::configureKernel(m_cm->maxDelay(), m_pitch32, m_pitch64);
		setStart();
	}
}



void
RuntimeData::stepSimulation(const std::vector<uint>& fstim)
		throw(DeviceAssertionFailure, std::logic_error)
{
	startSimulation(); // only has effect on first cycle

	/* A 32-bit counter can count up to around 4M seconds which is around 1200
	 * hours or 50 days */
	//! \todo use a 64-bit counter instead
	if(m_cycle == ~0U) {
		throw std::overflow_error("Cycle counter overflow");
	}
	m_cycle += 1;

	uint32_t* d_fstim = setFiringStimulus(fstim);
	uint32_t* d_fout = m_firingOutput->step();
	::stepSimulation(
			m_partitionCount,
			usingStdp(),
			m_cycle,
			m_recentFiring->deviceData(),
			m_neurons->deviceData(),
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
RuntimeData::applyStdp(float reward)
{
	if(deviceDirty()) {
		//! \todo issue a warning here?
		return; // we haven't even started simulating yet
	}

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
				stdpFn,
				reward);
	}

	m_deviceAssertions->check(m_cycle);
}


void
RuntimeData::printCycleCounters()
{
	std::ofstream outfile;
	outfile.open("cc.dat");
	m_cycleCounters->printCounters(outfile);
	outfile.close();
}


uint
RuntimeData::readFiring(
		const std::vector<uint>** cycles,
		const std::vector<uint>** nidx)
{
	return m_firingOutput->readFiring(cycles, nidx);
}


void
RuntimeData::flushFiringBuffer()
{
	m_firingOutput->flushBuffer();
}

}
