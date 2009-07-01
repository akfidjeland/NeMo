#include "RuntimeData.hpp"

extern "C" {
#include "kernel.h"
}
#include "L1SpikeQueue.hpp"
#include "FiringProbe.hpp"
#include "ConnectivityMatrix.hpp"
#include "CycleCounters.hpp"
#include "ThalamicInput.hpp"
#include "util.h"
#include "log.hpp"

#include <vector>
#include <assert.h>
#include <sys/time.h>


RuntimeData::RuntimeData(
		size_t partitionCount,
		size_t maxPartitionSize,
        uint maxDelay,
		size_t maxL0SynapsesPerDelay,
		size_t maxL0RevSynapsesPerDelay,
		size_t maxL1SynapsesPerDelay,
		size_t maxL1RevSynapsesPerDelay,
		//! \todo determine the entry size inside allocator
		size_t l1SQEntrySize,
		unsigned int maxReadPeriod) :
	maxPartitionSize(maxPartitionSize),
	partitionCount(partitionCount),
    m_maxDelay(maxDelay),
	m_cm(CM_COUNT, (ConnectivityMatrix*) NULL),
	m_pitch32(0),
	m_deviceDirty(true),
	m_usingSTDP(false),
	m_haveL1Connections(partitionCount != 1 && l1SQEntrySize != 0)
{
	spikeQueue = new L1SpikeQueue(partitionCount, l1SQEntrySize, maxL1SynapsesPerDelay);
	firingProbe = new FiringProbe(partitionCount, maxPartitionSize, maxReadPeriod);

	recentFiring = new NVector<uint32_t>(partitionCount, maxPartitionSize, false, 2);
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
			maxL0RevSynapsesPerDelay);

	m_cm[CM_L1] = new ConnectivityMatrix(
			partitionCount,
			maxPartitionSize,
			maxDelay,
			maxL1SynapsesPerDelay,
			maxL1RevSynapsesPerDelay);

    setPitch();

	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&m_deviceProperties, device);

	cycleCounters = new CycleCounters(partitionCount, m_deviceProperties.clockRate);
}



RuntimeData::~RuntimeData()
{
	delete spikeQueue;
	delete firingProbe;
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


void
RuntimeData::moveToDevice()
{
	if(m_deviceDirty) {
		neuronParameters->moveToDevice();
		for(std::vector<ConnectivityMatrix*>::iterator i = m_cm.begin();
			i != m_cm.end(); ++i) {
			(*i)->moveToDevice();
		}
        thalamicInput->moveToDevice();
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



void
checkPitch(size_t expected, size_t found)
{
	if(expected != found) {
		ERROR("RuntimeData::checkPitch: pitch mismatch in device memory allocation. Found %d, expected %d\n",
				(int) found, (int) expected);
	}
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
RuntimeData::setPitch()
{
    m_pitch32 = neuronParameters->wordPitch();
    checkPitch(m_pitch32, recentFiring->wordPitch());
    checkPitch(m_pitch32, thalamicInput->wordPitch());
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
timevalToMs(struct timeval& tv)
{
    return 1000 * tv.tv_sec + tv.tv_usec / 1000;
}

long
RuntimeData::elapsed()
{
    syncSimulation(this);
    gettimeofday(&m_end, NULL);
    struct timeval m_res;
    timersub(&m_end, &m_start, &m_res);
    return timevalToMs(m_res);
}


void
RuntimeData::setStart()
{
    gettimeofday(&m_start, NULL);
}




//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


bool
RuntimeData::usingSTDP() const
{
	return m_usingSTDP;
}


/* We just store the parameters here for the time being. The kernel is
 * configured on first launch */ 
void
RuntimeData::enableSTDP(int tauP, int tauD,
		float* potentiation,
		float* depression,
		float maxWeight)
{
	m_usingSTDP = true;
	m_stdpTauP = tauP;
	m_stdpTauD = tauD;

	//! \todo do more sensible error reporting here
	m_stdpPotentiation.resize(MAX_STDP_DELAY, 0.0f);
	if(tauP > MAX_STDP_DELAY) {
		fprintf(stderr, "Time window for potentiation (%u) exceeds CUDA backend maximum (%u)\n",
			tauP, MAX_STDP_DELAY);
		tauP = MAX_STDP_DELAY;
	}
	std::copy(potentiation, potentiation+tauP, m_stdpPotentiation.begin());

	m_stdpDepression.resize(MAX_STDP_DELAY, 0.0f);
	if(tauD > MAX_STDP_DELAY) {
		fprintf(stderr, "Time window for depression (%u) exceeds CUDA backend maximum (%u)\n",
			tauD, MAX_STDP_DELAY);
		tauP = MAX_STDP_DELAY;
	}
	std::copy(depression, depression+tauD, m_stdpDepression.begin());
	m_stdpMaxWeight = maxWeight;
}


float
RuntimeData::stdpMaxWeight() const
{
    return m_stdpMaxWeight;
}




/* The external API of the kernel is C-based, so we need wrappers to modify the
 * runtime data via the API. The headers are in kernel.h */


RTDATA
allocRuntimeData(
		size_t partitionCount,
		size_t maxPartitionSize,
        uint maxDelay,
		size_t maxL0SynapsesPerDelay,
		size_t maxL0RevSynapsesPerDelay,
		size_t maxL1SynapsesPerDelay,
		size_t maxL1RevSynapsesPerDelay,
		//! \todo determine the entry size inside allocator
		size_t l1SQEntrySize,
		uint maxReadPeriod)
{
	return new RuntimeData(
			partitionCount,
			maxPartitionSize,
            maxDelay,
			maxL0SynapsesPerDelay,
			maxL0RevSynapsesPerDelay,
			maxL1SynapsesPerDelay,
			maxL1RevSynapsesPerDelay,
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


void
setCMDRow(RTDATA rtdata,
		size_t cmIdx,
        unsigned int sourcePartition,
        unsigned int sourceNeuron,
        unsigned int delay,
        float* h_weights,
        unsigned int* h_targetPartition,
        unsigned int* h_targetNeuron,
        size_t length)
{
    rtdata->cm(cmIdx)->setRow(
        sourcePartition,
        sourceNeuron,
        delay,
        h_weights,
        h_targetPartition,
        h_targetNeuron,
        length);
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

size_t
readFiring(RTDATA rtdata,
		uint** cycles,
		uint** partitionIdx,
		uint** neuronIdx)
{
	size_t nfired;
	rtdata->firingProbe->readFiring(cycles, partitionIdx, neuronIdx, &nfired);
	return nfired;
}


//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


void
enableSTDP(RTDATA rtdata,
		int tauP, int tauD,
		float* potentiation,
		float* depression,
		float maxWeight)
{
	rtdata->enableSTDP(tauP, tauD, potentiation, depression, maxWeight);
}
