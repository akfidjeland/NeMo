#include "ConnectivityMatrix.hpp"

#include <cutil.h>
#include "log.hpp"
#include <algorithm>

#include "connectivityMatrix.cu_h"


ConnectivityMatrix::ConnectivityMatrix(
        size_t partitionCount,
        size_t maxPartitionSize,
		size_t maxDelay,
		size_t maxSynapsesPerDelay,
		size_t maxRevSynapsesPerDelay) :
	m_synapses(partitionCount,
			maxPartitionSize,
			maxDelay,
			maxSynapsesPerDelay,
			true,
			CM_SUBMATRICES),
    m_delayBits(partitionCount, maxPartitionSize, true),
    m_partitionCount(partitionCount),
    m_maxPartitionSize(maxPartitionSize),
    m_maxDelay(maxDelay),
	m_maxSynapsesPerDelay(partitionCount, 0),
	m_maxReverseSynapsesPerDelay(partitionCount, 0),
	m_reverse(partitionCount,
			maxPartitionSize,
			maxDelay,
			maxRevSynapsesPerDelay,
			true,
			RCM_SUBMATRICES),
	m_arrivalBits(partitionCount, maxPartitionSize, true)
{
	//! \todo this initialisation only needed as long as we use delay-specific reverse connectivity
	m_reverse.fillHostBuffer(INVALID_REVERSE_SYNAPSE, RCM_ADDRESS);
}



uint32_t*
ConnectivityMatrix::deviceDelayBits() const
{
	return m_delayBits.deviceData();
}


uint32_t*
ConnectivityMatrix::arrivalBits() const
{
	return m_arrivalBits.deviceData();
}



uint*
ConnectivityMatrix::deviceSynapsesD() const
{
	return m_synapses.deviceData();
}


uint*
ConnectivityMatrix::reverseConnectivity() const
{
	return m_reverse.deviceData();
}



size_t
ConnectivityMatrix::synapsePitchD() const
{
	return m_synapses.delayPitch();
}


size_t 
ConnectivityMatrix::submatrixSize() const
{
	return m_synapses.size();
}


size_t
ConnectivityMatrix::reverseSubmatrixSize() const
{
	return m_reverse.size();
}


size_t
ConnectivityMatrix::reversePitch() const
{
	return m_reverse.delayPitch();
}


/* Set row in delay-partitioned matrix */
void
ConnectivityMatrix::setDRow(
        uint sourcePartition,
        uint sourceNeuron,
        uint delay,
        const float* weights,
        const uint* targetPartition,
        const uint* targetNeuron,
        size_t length)
{
    if(length == 0)
        return;

	if(sourcePartition >= m_partitionCount) {
		ERROR("source partition index out of range");
	}

	if(sourceNeuron >= m_maxPartitionSize) {
		ERROR("source neuron index out of range");
	}

	if(delay > m_maxDelay || delay == 0) {
		ERROR("delay (%u) out of range (1-%u)", delay, m_maxDelay);
	}

    //! \todo allocate this only once!
	std::vector<uint> abuf(m_synapses.delayPitch(), 0);
	std::vector<uint> wbuf(m_synapses.delayPitch(), 0);

    bool setReverse = m_reverse.delayPitch() > 0;

	for(size_t i=0; i<length; ++i) {
		// see connectivityMatrix.cu_h for encoding format
		if(setReverse && weights[i] > 0.0f) { // only do STDP for excitatory synapses
			size_t rlen = m_reverse.addSynapse(
					targetPartition[i],
					targetNeuron[i],
					delay,
					packReverseSynapse(sourcePartition, sourceNeuron, i));
			m_maxReverseSynapsesPerDelay[targetPartition[i]] =
				std::max(m_maxReverseSynapsesPerDelay[targetPartition[i]], (uint) rlen);
			uint32_t arrivalBits = m_arrivalBits.getNeuron(targetPartition[i], targetNeuron[i]);
			arrivalBits |= 0x1 << (delay-1);
			m_arrivalBits.setNeuron(targetPartition[i], targetNeuron[i], arrivalBits);
		}
		wbuf[i] = reinterpret_cast<const uint32_t&>(weights[i]);
		abuf[i] = packSynapse(targetPartition[i], targetNeuron[i]);
	}

	m_synapses.setDelayRow(sourcePartition, sourceNeuron, delay, abuf, CM_ADDRESS);
	m_synapses.setDelayRow(sourcePartition, sourceNeuron, delay, wbuf, CM_WEIGHT);

	uint32_t delayBits = m_delayBits.getNeuron(sourcePartition, sourceNeuron);
	delayBits |= 0x1 << (delay-1);
	m_delayBits.setNeuron(sourcePartition, sourceNeuron, delayBits);

	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxSynapsesPerDelay[sourcePartition] =
		std::max(m_maxSynapsesPerDelay[sourcePartition], (uint) length);
}


void
ConnectivityMatrix::moveToDevice()
{
	m_delayBits.moveToDevice();
	/* The forward connectivity is retained as the address and weight data are
	 * needed if we do STDP tracing (along with the trace matrix itself). */
	m_synapses.copyToDevice();
	m_arrivalBits.moveToDevice();
	m_reverse.moveToDevice();
}



const std::vector<uint>&
ConnectivityMatrix::maxSynapsesPerDelay() const
{
	return m_maxSynapsesPerDelay;
}


const std::vector<uint>&
ConnectivityMatrix::maxReverseSynapsesPerDelay() const
{
	return m_maxReverseSynapsesPerDelay;
}



void
ConnectivityMatrix::clearSTDPTrace()
{
    m_synapses.fillDeviceBuffer(0, CM_STDP_TRACE);
}

//! \todo use this type in connectivityMatrix.cu
typedef union
{
    uint32_t dword_value;
    float float_value;
} synapse_t;


void
ConnectivityMatrix::printSTDPTrace()
{
    m_synapses.copyToHost(CM_STDP_TRACE);
    for(uint sourcePartition=0; sourcePartition<m_partitionCount; ++sourcePartition) {
        //! \todo could speed up traversal by using delay bits and max pitch
        for(uint sourceNeuron=0; sourceNeuron<m_maxPartitionSize; ++sourceNeuron) {
            for(uint delay=1; delay<=m_maxDelay; ++delay) {
                size_t pitch = m_synapses.delayPitch();
                //for(uint synapseIdx=0; synapseIdx<m_maxSynapsesPerDelay[sourcePartition]; ++synapseIdx)
                for(uint synapseIdx=0; synapseIdx<pitch; ++synapseIdx) 
                {
                    synapse_t w_tmp;
                    w_tmp.dword_value = m_synapses.h_lookup(sourcePartition, sourceNeuron, delay,
                            synapseIdx, CM_STDP_TRACE);
                    float w = w_tmp.float_value;
                    if(w != 0.0f) {
                        uint synapse = m_synapses.h_lookup(sourcePartition,
                                        sourceNeuron, delay, synapseIdx, CM_ADDRESS);
                        fprintf(stderr, "STDP: weight[%u-%u -> %u-%u] = %f\n",
                                sourcePartition, sourceNeuron,
                                targetPartition(synapse), targetNeuron(synapse), w);
                    }
                }
            }
        }
    }
}
