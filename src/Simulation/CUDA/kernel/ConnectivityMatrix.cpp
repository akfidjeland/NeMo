#include "ConnectivityMatrix.hpp"

#include <cutil.h>
#include "log.hpp"
#include <algorithm>

#include "connectivityMatrix.cu_h"

const int ConnectivityMatrix::InvalidNeuron = -1;

ConnectivityMatrix::ConnectivityMatrix(
        size_t partitionCount,
        size_t maxPartitionSize,
		size_t maxDelay,
		size_t maxSynapsesPerDelay,
		size_t maxRevSynapsesPerNeuron) :
	m_fsynapses(partitionCount,
			maxPartitionSize,
			maxDelay,
			maxSynapsesPerDelay,
			true,
			FCM_SUBMATRICES),
    m_delayBits(partitionCount, maxPartitionSize, true),
    m_partitionCount(partitionCount),
    m_maxPartitionSize(maxPartitionSize),
    m_maxDelay(maxDelay),
	m_maxSynapsesPerDelay(partitionCount, 0),
	m_rsynapses(partitionCount,
			maxPartitionSize,
			maxRevSynapsesPerNeuron),
	mf_targetPartition(
			partitionCount * maxPartitionSize * maxDelay * m_fsynapses.delayPitch(),
			ConnectivityMatrix::InvalidNeuron),
	mf_targetNeuron(
			partitionCount * maxPartitionSize * maxDelay * m_fsynapses.delayPitch(),
			ConnectivityMatrix::InvalidNeuron)
{ }



uint32_t*
ConnectivityMatrix::df_delayBits() const
{
	return m_delayBits.deviceData();
}



uint*
ConnectivityMatrix::df_synapses() const
{
	return m_fsynapses.d_data();
}


uint*
ConnectivityMatrix::dr_synapses() const
{
	return m_rsynapses.d_data();
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
	return m_rsynapses.pitch();
}


void
ConnectivityMatrix::setRow(
        uint sourcePartition,
        uint sourceNeuron,
        uint delay,
        const float* weights,
        const uint* targetPartition,
        const uint* targetNeuron,
        size_t f_length)
{
    if(f_length == 0)
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

	bool setReverse = !m_rsynapses.empty();

	for(size_t i=0; i<f_length; ++i) {
		// see connectivityMatrix.cu_h for encoding format
		if(setReverse && weights[i] > 0.0f) { // only do STDP for excitatory synapses
			m_rsynapses.addSynapse(
					sourcePartition, sourceNeuron, i,
					targetPartition[i], targetNeuron[i],
					delay);
		}
		wbuf[i] = reinterpret_cast<const uint32_t&>(weights[i]);
		abuf[i] = f_packSynapse(targetPartition[i], targetNeuron[i]);
	}

	m_fsynapses.setDelayRow(sourcePartition, sourceNeuron, delay, abuf, FCM_ADDRESS);
	m_fsynapses.setDelayRow(sourcePartition, sourceNeuron, delay, wbuf, FCM_WEIGHT);

	uint32_t delayBits = m_delayBits.getNeuron(sourcePartition, sourceNeuron);
	delayBits |= 0x1 << (delay-1);
	m_delayBits.setNeuron(sourcePartition, sourceNeuron, delayBits);

	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxSynapsesPerDelay[sourcePartition] =
		std::max(m_maxSynapsesPerDelay[sourcePartition], (uint) f_length);

	{
		size_t offset = m_fsynapses.offset(sourcePartition, sourceNeuron, delay, 0);
		std::copy(targetPartition,
				targetPartition + f_length,
				mf_targetPartition.begin() + offset);
		std::copy(targetNeuron,
				targetNeuron + f_length,
				mf_targetNeuron.begin() + offset);
	}
}


void
ConnectivityMatrix::moveToDevice()
{
	m_delayBits.moveToDevice();
	/* The forward connectivity is retained as the address and weight data are
	 * needed if we do STDP tracing (along with the trace matrix itself). */
	m_fsynapses.copyToDevice();
	m_rsynapses.moveToDevice();
}



void
ConnectivityMatrix::copyToHost(
				int* f_targetPartitions[],
				int* f_targetNeurons[],
				float* f_weights[],
				size_t* pitch)
{
	*pitch = m_fsynapses.delayPitch();
	if(mf_weights.empty()){
		mf_weights.resize(m_fsynapses.size(), 0.0f);
	}
	m_fsynapses.copyToHost(FCM_WEIGHT, mf_weights);
	*f_targetPartitions = &mf_targetPartition[0];
	*f_targetNeurons = &mf_targetNeuron[0];
	assert(sizeof(float) == sizeof(uint));
	*f_weights = (float*) &mf_weights[0];
}



const std::vector<uint>&
ConnectivityMatrix::f_maxSynapsesPerDelay() const
{
	return m_maxSynapsesPerDelay;
}



const std::vector<uint>&
ConnectivityMatrix::r_maxPartitionPitch() const
{
	return m_rsynapses.maxPartitionPitch();
}


//! \todo use this type in connectivityMatrix.cu
typedef union
{
    uint32_t dword_value;
    float float_value;
} synapse_t;


#if 0
void
ConnectivityMatrix::printSTDPTrace()
{
    m_fsynapses.copyToHost(FCM_STDP_TRACE);
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
                            synapseIdx, FCM_STDP_TRACE);
                    float w = w_tmp.float_value;
                    if(w != 0.0f) {
                        uint synapse = m_fsynapses.h_lookup(sourcePartition,
                                        sourceNeuron, delay, synapseIdx, FCM_ADDRESS);
                        fprintf(stderr, "STDP: weight[%u-%u -> %u-%u] = %f\n",
                                sourcePartition, sourceNeuron,
                                targetPartition(synapse), targetNeuron(synapse), w);
                    }
                }
            }
        }
    }
}
#endif



void
ConnectivityMatrix::df_clear(size_t plane)
{
	m_fsynapses.d_fill(0, plane);
}


void
ConnectivityMatrix::dr_clear(size_t plane)
{
	m_rsynapses.d_fill(plane, 0);
}
