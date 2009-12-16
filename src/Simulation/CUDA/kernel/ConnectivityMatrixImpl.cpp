#include "ConnectivityMatrixImpl.hpp"

#include "util.h"
#include "log.hpp"
#include "RSMatrix.hpp"
#include "SynapseGroup.hpp"
#include "connectivityMatrix.cu_h"
#include "dispatchTable.cu_h"

#include <algorithm>
#include <stdexcept>


const int ConnectivityMatrixImpl::InvalidNeuron = -1;

ConnectivityMatrixImpl::ConnectivityMatrixImpl(
        size_t partitionCount,
        size_t maxPartitionSize,
		size_t maxDelay,
		size_t maxSynapsesPerDelay,
		bool setReverse) :
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
	m_setReverse(setReverse),
	mf_targetPartition(
			partitionCount * maxPartitionSize * maxDelay * m_fsynapses.delayPitch(),
			ConnectivityMatrixImpl::InvalidNeuron),
	mf_targetNeuron(
			partitionCount * maxPartitionSize * maxDelay * m_fsynapses.delayPitch(),
			ConnectivityMatrixImpl::InvalidNeuron)
{
	for(uint p = 0; p < partitionCount; ++p) {
		m_rsynapses.push_back(new RSMatrix(maxPartitionSize));
	}
}



uint64_t*
ConnectivityMatrixImpl::df_delayBits() const
{
	return m_delayBits.deviceData();
}



uint*
ConnectivityMatrixImpl::df_synapses() const
{
	return m_fsynapses.d_data();
}


size_t
ConnectivityMatrixImpl::df_pitch() const
{
	return m_fsynapses.delayPitch();
}


size_t 
ConnectivityMatrixImpl::df_planeSize() const
{
	return m_fsynapses.size();
}


void
ConnectivityMatrixImpl::setRow(
		uint sourcePartition,
		uint sourceNeuron,
		uint delay,
		const float* weights,
		const uint* targetPartition,
		const uint* targetNeuron,
		const uint* isPlastic,
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

	for(size_t i=0; i<f_length; ++i) {
		// see connectivityMatrix.cu_h for encoding format
		if(m_setReverse && isPlastic[i]) {
			m_rsynapses[targetPartition[i]]->addSynapse(
					sourcePartition, sourceNeuron, i,
					targetNeuron[i], delay);
		}
		wbuf[i] = reinterpret_cast<const uint32_t&>(weights[i]);
		abuf[i] = f_packSynapse(targetPartition[i], targetNeuron[i]);
	}

	m_fsynapses.setDelayRow(sourcePartition, sourceNeuron, delay, abuf, FCM_ADDRESS);
	m_fsynapses.setDelayRow(sourcePartition, sourceNeuron, delay, wbuf, FCM_WEIGHT);

	m_fsynapses2[nemo::ForwardIdx(sourcePartition, delay)].addSynapses(
			sourceNeuron,
			f_length,
			targetPartition,
			targetNeuron,
			weights);

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
ConnectivityMatrixImpl::moveToDevice(bool isL0)
{
	m_delayBits.moveToDevice();
	/* The forward connectivity is retained as the address and weight data are
	 * needed if we do STDP tracing (along with the trace matrix itself). */
	m_fsynapses.copyToDevice();
	for(uint p=0; p < m_partitionCount; ++p){
		m_rsynapses[p]->moveToDevice();
	}

	for(fcm_t::iterator i = m_fsynapses2.begin();
			i != m_fsynapses2.end(); ++i) {
		i->second.moveToDevice();
	}

	f_setDispatchTable(isL0);
}



void
ConnectivityMatrixImpl::copyToHost(
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
ConnectivityMatrixImpl::f_maxSynapsesPerDelay() const
{
	return m_maxSynapsesPerDelay;
}



//! \todo use this type in connectivityMatrix.cu
typedef union
{
    uint32_t dword_value;
    float float_value;
} synapse_t;


#if 0
void
ConnectivityMatrixImpl::printSTDPTrace()
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
ConnectivityMatrixImpl::df_clear(size_t plane)
{
	m_fsynapses.d_fill(0, plane);
}



void
ConnectivityMatrixImpl::clearStdpAccumulator()
{
	//! \todo this might be done better in a single kernel, to reduce bus traffic
	for(uint p=0; p < m_partitionCount; ++p){
		m_rsynapses[p]->clearStdpAccumulator();
	}
}



size_t
ConnectivityMatrixImpl::d_allocated() const
{
	size_t rcm = 0;
	for(std::vector<RSMatrix*>::const_iterator i = m_rsynapses.begin();
			i != m_rsynapses.end(); ++i) {
		rcm += (*i)->d_allocated();
	}

	return m_fsynapses.d_allocated()
		+ m_delayBits.d_allocated()
		+ rcm;
}



/* Pack a device pointer to a 32-bit value */
//! \todo replace with non-template version
template<typename T>
DEVICE_UINT_PTR_T
devicePointer(T ptr)
{
	uint64_t ptr64 = (uint64_t) ptr;
#ifndef __DEVICE_EMULATION__
	//! \todo: look up this data at runtime
	//! \todo assert that we can fit all device addresses in 32b address.
	const uint64_t MAX_ADDRESS = 4294967296LL; // on device
	if(ptr64 >= MAX_ADDRESS) {
		throw std::range_error("Device pointer larger than 32 bits");
	}
#endif
	return (DEVICE_UINT_PTR_T) ptr64;

}



/* Map function over vector of reverse synapse matrix */
template<typename T, class S>
const std::vector<DEVICE_UINT_PTR_T>
mapDevicePointer(const std::vector<S*>& vec, std::const_mem_fun_t<T, S> fun)
{
	std::vector<DEVICE_UINT_PTR_T> ret(vec.size(), 0);
	for(uint p=0; p < vec.size(); ++p){
		T ptr = fun(vec[p]);
		ret[p] = devicePointer(ptr);
	}
	return ret;
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrixImpl::r_partitionPitch() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::pitch));
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrixImpl::r_partitionAddress() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_address));
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrixImpl::r_partitionStdp() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_stdp));
}



void
ConnectivityMatrixImpl::f_setDispatchTable(bool isL0)
{
	//! \todo remove magic
	size_t delayCount = 64;

	size_t width = m_partitionCount;
	size_t height = delayCount;
	size_t size = width * height;

	fcm_ref_t null = fcm_packReference(0, 0);
	std::vector<fcm_ref_t> table(size, null);

	for(fcm_t::const_iterator i = m_fsynapses2.begin();
			i != m_fsynapses2.end(); ++i) {

		nemo::ForwardIdx fidx = i->first;
		const SynapseGroup& sg = i->second;

		// x: delay, y : partition
		size_t addr = fidx.source * delayCount + (fidx.delay-1);

		void* fcm_addr = sg.d_address();
		size_t fcm_pitch = sg.wpitch();
		table.at(addr) = fcm_packReference(fcm_addr, fcm_pitch);
	}

	cudaArray* f_dispatch;
	if(isL0) {
		f_dispatch = ::f0_setDispatchTable(m_partitionCount, delayCount, table);
	} else {
		f_dispatch = ::f1_setDispatchTable(m_partitionCount, delayCount, table);
	}
	mf_dispatch = boost::shared_ptr<cudaArray>(f_dispatch, cudaFreeArray);
}
