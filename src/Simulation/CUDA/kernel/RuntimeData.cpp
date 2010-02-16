#include "RuntimeData.hpp"

extern "C" {
#include "kernel.h"
}
#include "FiringOutput.hpp"
#include "ConnectivityMatrix.hpp"
#include "CycleCounters.hpp"
#include "NeuronParameters.hpp"
#include "ThalamicInput.hpp"
#include "util.h"
#include "log.hpp"
#include "fixedpoint.hpp"
#include "bitvector.hpp"

#include <vector>
#include <assert.h>



RuntimeData::RuntimeData(
		size_t maxPartitionSize,
		bool setReverse,
		unsigned int maxReadPeriod) :
	firingOutput(NULL),
	recentFiring(NULL),
	firingStimulus(NULL),
	thalamicInput(NULL),
	maxPartitionSize(maxPartitionSize),
	cycleCounters(NULL),
	m_neurons(new NeuronParameters(maxPartitionSize)),
	m_cm(new ConnectivityMatrix(maxPartitionSize, setReverse)),
	m_pitch32(0),
	m_pitch64(0),
	m_partitionCount(0),
	m_deviceDirty(true),
	m_maxReadPeriod(maxReadPeriod)
{

	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&m_deviceProperties, device);
}



RuntimeData::~RuntimeData()
{
	if(firingOutput) delete firingOutput;
	if(recentFiring) delete recentFiring;
	if(firingStimulus) delete firingStimulus;
	if(thalamicInput) delete thalamicInput;
	if(cycleCounters) delete cycleCounters;
	delete m_cm;
	delete m_neurons;
}



uint
RuntimeData::maxDelay() const
{
	return m_cm->maxDelay();
}



extern void
configureStdp(
		uint preFireWindow,
		uint postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		weight_dt* stdpFn);



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
		firingOutput = new FiringOutput(m_partitionCount, maxPartitionSize, m_maxReadPeriod);
		recentFiring = new NVector<uint64_t>(m_partitionCount, maxPartitionSize, false, 2);
		//! \todo seed properly from outside function
		thalamicInput = new ThalamicInput(m_partitionCount, maxPartitionSize, 0);
		m_neurons->setSigma(*thalamicInput);
		thalamicInput->moveToDevice();
		cycleCounters = new CycleCounters(m_partitionCount, m_deviceProperties.clockRate);
		firingStimulus = new NVector<uint32_t>(m_partitionCount, BV_WORD_PITCH, false);

		setPitch();
	    m_deviceDirty = false;
	}
}


float*
RuntimeData::d_neurons() const
{
	return m_neurons->deviceData();
}



size_t
RuntimeData::neuronVectorLength() const
{
	assert(m_neurons);
	return m_neurons->d_vectorLength();
}



bool
RuntimeData::deviceDirty() const
{
	return m_deviceDirty;
}



struct ConnectivityMatrix*
RuntimeData::cm() const
{
	return m_cm;
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
RuntimeData::setFiringStimulus(size_t count, const int* nidx)
{
	if(count == 0) 
		return NULL;

	//! \todo use internal host buffer with pinned memory instead
	size_t pitch = firingStimulus->wordPitch();
	std::vector<uint32_t> hostArray(firingStimulus->size(), 0);

	for(size_t i=0; i < count; ++i){
		//! \todo share this translation with NeuronParameters and CMImpl
		size_t nn = nidx[i] % maxPartitionSize;
		size_t pn = nidx[i] / maxPartitionSize;
		//! \todo should check the size of this particular partition
		assert(nn < maxPartitionSize );
		assert(pn < m_partitionCount);
		size_t word = pn * pitch + nn / 32;
		size_t bit = nn % 32;
		hostArray[word] |= 1 << bit;
	}

	CUDA_SAFE_CALL(cudaMemcpy(
				firingStimulus->deviceData(),
				&hostArray[0],
				m_partitionCount * firingStimulus->bytePitch(),
				cudaMemcpyHostToDevice));

	return firingStimulus->deviceData();
}



size_t
RuntimeData::pitch32() const
{
    return m_pitch32;
}



size_t
RuntimeData::pitch64() const
{
    return m_pitch64;
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
	total += firingStimulus   ? firingStimulus->d_allocated()   : 0;
	total += recentFiring     ? recentFiring->d_allocated()     : 0;
	total += m_neurons        ? m_neurons->d_allocated() : 0;
	total += firingOutput     ? firingOutput->d_allocated()     : 0;
	total += thalamicInput    ? thalamicInput->d_allocated()    : 0;
	total += m_cm             ? m_cm->d_allocated()             : 0;
	return total;
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
RuntimeData::setPitch()
{
	size_t pitch1 = firingStimulus->wordPitch();
	m_pitch32 = m_neurons->wordPitch();
	m_pitch64 = recentFiring->wordPitch();
	//! \todo fold thalamic input into neuron parameters
	checkPitch(m_pitch32, thalamicInput->wordPitch());
	checkPitch(pitch1, firingOutput->wordPitch());
	bv_setPitch(pitch1);
}



void
RuntimeData::step()
{
    m_cycle += 1;
}



uint32_t
RuntimeData::cycle() const
{
    return m_cycle;
}


//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


long
RuntimeData::elapsed()
{
    syncSimulation(this);
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



//-----------------------------------------------------------------------------
// External API
//-----------------------------------------------------------------------------


/* The external API of the kernel is C-based, so we need wrappers to modify the
 * runtime data via the API. The headers are in kernel.h */


RTDATA
allocRuntimeData(
		size_t maxPartitionSize,
		uint setReverse,
		uint maxReadPeriod)
{
	return new RuntimeData(maxPartitionSize, (bool) setReverse, maxReadPeriod);
}


void
freeRuntimeData(RTDATA mem)
{
	delete mem;
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
addNeuron(RTDATA rt,
		unsigned int idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	rt->addNeuron(idx, a, b, c, d, u, v, sigma);
}


size_t
allocatedDeviceMemory(RTDATA rt)
{
	return rt->d_allocated();
}



//-----------------------------------------------------------------------------
// Connectivity matrix
//-----------------------------------------------------------------------------


void
addSynapses(RTDATA rtdata,
		unsigned int source,
		unsigned int targets[],
		unsigned int delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length)
{
	rtdata->cm()->setRow(
		source,
		targets,
		delays,
		weights,
		is_plastic,
		length);
}



void
setCMDRow(RTDATA rtdata,
		unsigned int sourceNeuron,
		unsigned int delay,
		unsigned int* targetNeuron,
		float* weights,
		unsigned char* isPlastic,
		size_t length)
{
	std::vector<unsigned int> delays(length, delay);
	rtdata->cm()->setRow(
		sourceNeuron,
		targetNeuron,
		&delays[0],
		weights,
		isPlastic,
		length);
}



size_t
getCMDRow(RTDATA rtdata,
		unsigned int sourcePartition,
		unsigned int sourceNeuron,
		unsigned int delay,
		unsigned int* targetPartition[],
		unsigned int* targetNeuron[],
		float* weights[],
		unsigned char* plastic[])
{
	return rtdata->cm()->getRow(sourcePartition, sourceNeuron, delay,
			rtdata->cycle(), targetPartition, targetNeuron, weights, plastic);
}




//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


void
printCycleCounters(RTDATA rtdata)
{
	rtdata->cycleCounters->printCounters();
}



long int
elapsedMs(RTDATA rtdata)
{
	return rtdata->elapsed();
}


void
resetTimer(RTDATA rtdata)
{
	// force all execution to complete first
	syncSimulation(rtdata);
	rtdata->setStart();
}


//-----------------------------------------------------------------------------
// Generated firing
//-----------------------------------------------------------------------------

void
readFiring(RTDATA rtdata,
		uint** cycles,
		uint** neuronIdx,
		uint* nfired,
		uint* ncycles)
{
	rtdata->firingOutput->readFiring(cycles, neuronIdx, nfired, ncycles);
}


void
flushFiringBuffer(RTDATA rtdata)
{
	rtdata->firingOutput->flushBuffer();
}

//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------



void
enableStdp(RTDATA rtdata,
		unsigned int pre_len,
		unsigned int post_len,
		float* pre_fn,
		float* post_fn,
		float w_max,
		float w_min)
{
	nemo::configure_stdp(rtdata->stdpFn, pre_len, post_len, pre_fn, post_fn, w_max, w_min);
}
