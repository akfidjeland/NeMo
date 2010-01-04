#include "RuntimeData.hpp"

extern "C" {
#include "kernel.h"
}
#include "L1SpikeQueue.hpp"
#include "FiringOutput.hpp"
#include "ConnectivityMatrix.hpp"
#include "CycleCounters.hpp"
#include "ThalamicInput.hpp"
#include "util.h"
#include "log.hpp"

#include <vector>
#include <assert.h>



RuntimeData::RuntimeData(
		size_t partitionCount,
		size_t maxPartitionSize,
        uint maxDelay,
		size_t maxL0SynapsesPerDelay,
		size_t maxL1SynapsesPerDelay,
		bool setReverse,
		//! \todo determine the entry size inside allocator
		size_t l1SQEntrySize,
		unsigned int maxReadPeriod) :
	maxPartitionSize(maxPartitionSize),
	partitionCount(partitionCount),
	m_maxDelay(maxDelay),
	m_cm(CM_COUNT, (ConnectivityMatrix*) NULL),
	m_pitch32(0),
	m_pitch64(0),
	m_deviceDirty(true),
	m_haveL1Connections(partitionCount != 1 && l1SQEntrySize != 0)
{
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&m_deviceProperties, device);

	spikeQueue = new L1SpikeQueue(partitionCount, l1SQEntrySize, maxL1SynapsesPerDelay);
	firingOutput = new FiringOutput(partitionCount, maxPartitionSize, maxReadPeriod);

	recentFiring = new NVector<uint64_t>(partitionCount, maxPartitionSize, false, 2);
	neuronParameters = new NVector<float>(partitionCount, maxPartitionSize, true, NVEC_COUNT);

	firingStimulus = new NVector<uint32_t>(
			partitionCount,
			DIV_CEIL(maxPartitionSize, 32),
			false);

	//! \todo seed properly from outside function
	thalamicInput = new ThalamicInput(partitionCount, maxPartitionSize, 0);

	m_cm[CM_L0] = new ConnectivityMatrix(
			partitionCount,
			maxPartitionSize,
			maxDelay,
			maxL0SynapsesPerDelay,
			setReverse);

	m_cm[CM_L1] = new ConnectivityMatrix(
			partitionCount,
			maxPartitionSize,
			maxDelay,
			maxL1SynapsesPerDelay,
			setReverse);

	setPitch();

	cycleCounters = new CycleCounters(partitionCount, m_deviceProperties.clockRate);
}



RuntimeData::~RuntimeData()
{
	delete spikeQueue;
	delete firingOutput;
	delete recentFiring;
	delete neuronParameters;
	delete firingStimulus;
	delete thalamicInput;
	delete cycleCounters;
	for(std::vector<ConnectivityMatrix*>::iterator i = m_cm.begin();
			i != m_cm.end(); ++i) {
		if(*i != NULL) {
			delete *i;
		}
	}
}



uint
RuntimeData::maxDelay() const
{
    return m_maxDelay;
}



extern void
configureStdp(
		uint preFireWindow,
		uint postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		float* stdpFn);


void
RuntimeData::moveToDevice()
{
	if(m_deviceDirty) {
		neuronParameters->moveToDevice();
		m_cm[CM_L0]->moveToDevice(true);
		m_cm[CM_L1]->moveToDevice(false);
		thalamicInput->moveToDevice();
		if(stdpFn.enabled()) {
			configureStdp(stdpFn.preFireWindow(),
					stdpFn.postFireWindow(),
					stdpFn.potentiationBits(),
					stdpFn.depressionBits(),
					const_cast<float*>(&stdpFn.function()[0]));
		}
	    m_deviceDirty = false;
	}
}



bool
RuntimeData::deviceDirty() const
{
	return m_deviceDirty;
}


bool
RuntimeData::haveL1Connections() const
{
	return m_haveL1Connections;
}




struct ConnectivityMatrix*
RuntimeData::cm(size_t idx) const
{
	if(idx >= CM_COUNT) {
		ERROR("RuntimeData::cm: invalid connectivity matrix (%u) requested", (uint) idx);
		return NULL;
	}
	return m_cm[idx];
}




/*! Copy firing stimulus from host to device. Array indices only tested in
 * debugging mode.
 * 
 * \param count
 *		Number of neurons whose firing should be forced
 * \param cidx
 * 		Cluster indices of neurons whose firing should be forced
 * \param nidx
 * 		Neuron indices (within cluster) of neurons whose firing should be forced
 * \param mem
 *		Data structure containing device and host buffers
 *
 * \return 
 *		Pointer to pass to kernel (which is NULL if there's no firing data).
 */
uint32_t*
RuntimeData::setFiringStimulus(
		size_t count,
		const int* pidx,
		const int* nidx)
{
	if(count == 0) 
		return NULL;

	//! \todo use internal host buffer with pinned memory instead
	size_t pitch = firingStimulus->wordPitch();
	std::vector<uint32_t> hostArray(firingStimulus->size(), 0);

	for(size_t i=0; i < count; ++i){
		size_t nn = nidx[i];
		size_t pn = pidx[i];
		//! \todo should check the size of this particular partition
		assert(nn < maxPartitionSize );
		assert(pn < partitionCount);
		size_t word = pn * pitch + nn / 32;
		size_t bit = nn % 32;
		hostArray[word] |= 1 << bit;
	}

	CUDA_SAFE_CALL(cudaMemcpy(
				firingStimulus->deviceData(),
				&hostArray[0],
				partitionCount * firingStimulus->bytePitch(),
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
	total += spikeQueue       ? spikeQueue->d_allocated()       : 0;
	total += firingStimulus   ? firingStimulus->d_allocated()   : 0;
	total += recentFiring     ? recentFiring->d_allocated()     : 0;
	total += neuronParameters ? neuronParameters->d_allocated() : 0;
	total += firingOutput     ? firingOutput->d_allocated()     : 0;
	total += thalamicInput    ? thalamicInput->d_allocated()    : 0;
	for(std::vector<ConnectivityMatrix*>::const_iterator i = m_cm.begin();
			i != m_cm.end(); ++i) {
		total += *i ?  (*i)->d_allocated() : 0;
	}
	return total;
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
RuntimeData::setPitch()
{
	m_pitch1 = firingStimulus->wordPitch();
	m_pitch32 = neuronParameters->wordPitch();
	m_pitch64 = recentFiring->wordPitch();
	//! \todo fold thalamic input into neuron parameters
	checkPitch(m_pitch32, thalamicInput->wordPitch());
	checkPitch(m_pitch1, firingOutput->wordPitch());
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
		size_t partitionCount,
		size_t maxPartitionSize,
		uint maxDelay,
		size_t maxL0SynapsesPerDelay,
		size_t maxL1SynapsesPerDelay,
		uint setReverse,
		//! \todo determine the entry size inside allocator
		size_t l1SQEntrySize,
		uint maxReadPeriod)
{
	return new RuntimeData(
			partitionCount,
			maxPartitionSize,
			maxDelay,
			maxL0SynapsesPerDelay,
			maxL1SynapsesPerDelay,
			(bool) setReverse,
			l1SQEntrySize,
			maxReadPeriod);
}


void
freeRuntimeData(RTDATA mem)
{
	delete mem;
}


void
loadParam(RTDATA rt,
        size_t paramIdx,
        size_t partitionIdx,
        size_t partitionSize,
        float* arr)
{
	rt->neuronParameters->setPartition(partitionIdx, arr, partitionSize, paramIdx);
}


size_t
allocatedDeviceMemory(RTDATA rt)
{
	return rt->d_allocated();
}


//-----------------------------------------------------------------------------
// Thalamic input
//-----------------------------------------------------------------------------


void
loadThalamicInputSigma(RTDATA rt,
		size_t partitionIdx,
		size_t partitionSize,
		float* arr)
{
	rt->thalamicInput->setSigma(partitionIdx, arr, partitionSize);
}


//-----------------------------------------------------------------------------
// Connectivity matrix
//-----------------------------------------------------------------------------


/*! \todo If care is not taken in writing the dense matrix, the kernel might
 * end up with shared memory bank conflicts. The reason is that, unlike in
 * dense mode, thread indices can point to arbitrary postsynaptic neurons. It
 * is thus possible to have several threads within a warp accessing a
 * postsynaptic neuron in the same bank. */


//! \todo change order (weight last) for consistency with getCMDRow
void
setCMDRow(RTDATA rtdata,
		size_t cmIdx,
		unsigned int sourcePartition,
		unsigned int sourceNeuron,
		unsigned int delay,
		float* weights,
		unsigned int* targetPartition,
		unsigned int* targetNeuron,
		unsigned char* isPlastic,
		size_t length)
{
	rtdata->cm(cmIdx)->setRow(
		sourcePartition,
		sourceNeuron,
		delay,
		weights,
		targetPartition,
		targetNeuron,
		isPlastic,
		length);
}



size_t
getCMDRow(RTDATA rtdata,
		size_t cmIdx,
		unsigned int sourcePartition,
		unsigned int sourceNeuron,
		unsigned int delay,
		unsigned int* targetPartition[],
		unsigned int* targetNeuron[],
		float* weights[],
		unsigned char* plastic[])
{
	return rtdata->cm(cmIdx)->getRow(sourcePartition, sourceNeuron, delay,
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
		uint** partitionIdx,
		uint** neuronIdx,
		uint* nfired,
		uint* ncycles)
{
	rtdata->firingOutput->readFiring(cycles, partitionIdx,
			neuronIdx, nfired, ncycles);
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
