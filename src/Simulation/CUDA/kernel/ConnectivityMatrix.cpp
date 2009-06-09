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
	m_fsynapses(partitionCount,
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
	m_rsynapses(partitionCount,
			maxPartitionSize,
			maxDelay,
			maxRevSynapsesPerDelay,
			true,
			RCM_SUBMATRICES),
	m_arrivalBits(partitionCount, maxPartitionSize, true)
{
	//! \todo this initialisation only needed as long as we use delay-specific reverse connectivity
	m_rsynapses.fillHostBuffer(INVALID_REVERSE_SYNAPSE, RCM_ADDRESS);
}



uint32_t*
ConnectivityMatrix::df_delayBits() const
{
	return m_delayBits.deviceData();
}


uint32_t*
ConnectivityMatrix::dr_delayBits() const
{
	return m_arrivalBits.deviceData();
}



uint*
ConnectivityMatrix::df_synapses() const
{
	return m_fsynapses.deviceData();
}


uint*
ConnectivityMatrix::dr_synapses() const
{
	return m_rsynapses.deviceData();
}



size_t
ConnectivityMatrix::df_pitch() const
{
	return m_fsynapses.delayPitch();
}


size_t 
ConnectivityMatrix::df_planeSize() const
{
	return m_fsynapses.size();
}


size_t
ConnectivityMatrix::dr_planeSize() const
{
	return m_rsynapses.size();
}


size_t
ConnectivityMatrix::dr_pitch() const
{
	return m_rsynapses.delayPitch();
}


void
ConnectivityMatrix::setRow(
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
	std::vector<uint> abuf(m_fsynapses.delayPitch(), 0);
	std::vector<uint> wbuf(m_fsynapses.delayPitch(), 0);

    bool setReverse = m_rsynapses.delayPitch() > 0;

	for(size_t i=0; i<length; ++i) {
		// see connectivityMatrix.cu_h for encoding format
		if(setReverse && weights[i] > 0.0f) { // only do STDP for excitatory synapses
			size_t rlen = m_rsynapses.addSynapse(
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

	m_fsynapses.setDelayRow(sourcePartition, sourceNeuron, delay, abuf, CM_ADDRESS);
	m_fsynapses.setDelayRow(sourcePartition, sourceNeuron, delay, wbuf, CM_WEIGHT);

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
	m_fsynapses.copyToDevice();
	m_arrivalBits.moveToDevice();
	m_rsynapses.moveToDevice();
}



const std::vector<uint>&
ConnectivityMatrix::f_maxSynapsesPerDelay() const
{
	return m_maxSynapsesPerDelay;
}


const std::vector<uint>&
ConnectivityMatrix::r_maxSynapsesPerDelay() const
{
	return m_maxReverseSynapsesPerDelay;
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
    m_fsynapses.copyToHost(CM_STDP_TRACE);
    for(uint sourcePartition=0; sourcePartition<m_partitionCount; ++sourcePartition) {
        //! \todo could speed up traversal by using delay bits and max pitch
        for(uint sourceNeuron=0; sourceNeuron<m_maxPartitionSize; ++sourceNeuron) {
            for(uint delay=1; delay<=m_maxDelay; ++delay) {
                size_t pitch = m_fsynapses.delayPitch();
                //for(uint synapseIdx=0; synapseIdx<m_maxSynapsesPerDelay[sourcePartition]; ++synapseIdx)
                for(uint synapseIdx=0; synapseIdx<pitch; ++synapseIdx) 
                {
                    synapse_t w_tmp;
                    w_tmp.dword_value = m_fsynapses.h_lookup(sourcePartition, sourceNeuron, delay,
                            synapseIdx, CM_STDP_TRACE);
                    float w = w_tmp.float_value;
                    if(w != 0.0f) {
                        uint synapse = m_fsynapses.h_lookup(sourcePartition,
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



void
ConnectivityMatrix::df_clear(size_t plane)
{
	//! \todo bounds checking
	m_fsynapses.fillDeviceBuffer(0, plane);
}


void
ConnectivityMatrix::dr_clear(size_t plane)
{
	//! \todo bounds checking
	m_rsynapses.fillDeviceBuffer(0, plane);
}
