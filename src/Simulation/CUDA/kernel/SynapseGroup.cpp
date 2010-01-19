//! \file SynapseGroup.cpp

#include "SynapseGroup.hpp"
#include "connectivityMatrix.cu_h"
#include "util.h"
#include "kernel.cu_h"


SynapseGroup::SynapseGroup() :
	md_bpitch(0),
	m_allocated(0),
	m_lastSync(-1)
{ }



sidx_t
SynapseGroup::addSynapse(
		nidx_t sourceNeuron,
		pidx_t partition,
		nidx_t neuron,
		float weight,
		uchar plastic)
{
	Row& row = mh_synapses[sourceNeuron];
	row.addresses.push_back(f_packSynapse(partition, neuron));
	row.weights.push_back(weight);

	assert(row.addresses.size() == row.weights.size());

	mf_targetPartition[sourceNeuron].push_back(partition);
	mf_targetNeuron[sourceNeuron].push_back(neuron);
	mf_plastic[sourceNeuron].push_back(plastic);

	return row.addresses.size() - 1;
}



//! \todo move Synapse type out of this class and pass in just a vector
void
SynapseGroup::addSynapses(
		nidx_t sourceNeuron,
		size_t ncount,
		const pidx_t partition[],
		const nidx_t neuron[],
		const float weight[],
		const uchar plastic[])
{
	Row& row = mh_synapses[sourceNeuron];
	for(size_t n = 0; n < ncount; ++n) {
		row.addresses.push_back(f_packSynapse(partition[n], neuron[n]));
		row.weights.push_back(weight[n]);
	}

	std::copy(partition, partition+ncount, back_inserter(mf_targetPartition[sourceNeuron]));
	std::copy(neuron, neuron+ncount, back_inserter(mf_targetNeuron[sourceNeuron]));
	std::copy(plastic, plastic+ncount, back_inserter(mf_plastic[sourceNeuron]));
	assert(row.addresses.size() == row.weights.size());
}



size_t
SynapseGroup::maxSynapsesPerNeuron() const
{
	size_t n = 0;
	for(std::map<nidx_t, Row>::const_iterator i = mh_synapses.begin();
			i != mh_synapses.end(); ++i) {
		n = std::max(n, i->second.addresses.size());
	}
	return n;
}



boost::shared_ptr<SynapseGroup::synapse_t>
SynapseGroup::moveToDevice()
{
	if(mh_synapses.empty()) {
		return boost::shared_ptr<SynapseGroup::synapse_t>();
	}

	/* Aligning pitch to warp size should have no negative impact on memory
	 * bandwidth, but can reduce thread divergence. On a network with 2k
	 * neurons with 2M synapses (1.8M L0, 0.2M L1) we find a small throughput
	 * improvement (from 61M spike deliveries per second to 63M). */
	size_t minWordPitch = maxSynapsesPerNeuron();
	size_t alignedWordPitch = ALIGN(minWordPitch, WARP_SIZE);
	size_t desiredBytePitch = alignedWordPitch * sizeof(synapse_t);
	size_t height = FCM_SUBMATRICES * MAX_PARTITION_SIZE;

	synapse_t* d_data = NULL;

	CUDA_SAFE_CALL(
			cudaMallocPitch((void**) &d_data,
				&md_bpitch,
				desiredBytePitch,
				height));
	m_allocated = md_bpitch * height;

	/* There's no need to clear to allocated memory, as we completely fill it
	 * below */

	size_t wordPitch = md_bpitch / sizeof(synapse_t);
	std::vector<synapse_t> h_data(wordPitch * height, 0); 

	synapse_t* astart = &h_data[0] + FCM_ADDRESS * MAX_PARTITION_SIZE * wordPitch;
	synapse_t* wstart = &h_data[0] + FCM_WEIGHT  * MAX_PARTITION_SIZE * wordPitch;

	for(std::map<nidx_t, Row>::const_iterator r = mh_synapses.begin();
			r != mh_synapses.end(); ++r) {

		nidx_t sourceNeuron = r->first;
		const Row row = r->second;
		size_t row_idx = sourceNeuron * wordPitch;

		assert(row.addresses.size() == row.weights.size());
		assert(row.addresses.size() * sizeof(synapse_t) <= md_bpitch);

		/*! note that std::copy won't work as it will silently cast floats to integers */
		memcpy(astart + row_idx, &row.addresses[0], row.addresses.size() * sizeof(synapse_t));
		memcpy(wstart + row_idx, &row.weights[0], row.weights.size() * sizeof(synapse_t));
	}

	CUDA_SAFE_CALL(cudaMemcpy(d_data, &h_data[0], m_allocated, cudaMemcpyHostToDevice));

	mh_synapses.clear();
	md_synapses = boost::shared_ptr<synapse_t>(d_data, cudaFree);	
	return md_synapses;
}



size_t
SynapseGroup::getWeights(
		nidx_t sourceNeuron,
		uint currentCycle,
		pidx_t* partition[],
		nidx_t* neuron[],
		weight_t* weight[],
		uchar* plastic[])
{
	if(mf_targetPartition.find(sourceNeuron) == mf_targetPartition.end()) {
		partition = NULL;
		neuron = NULL;
		weight = NULL;
		return 0;
	}

	size_t w_planeSize = planeSize() / sizeof(synapse_t);

	if(mf_weights.empty()) {
		mf_weights.resize(w_planeSize, 0);
	}

	if(currentCycle != m_lastSync) {
		// if we haven't already synced this cycle, do so now
		CUDA_SAFE_CALL(cudaMemcpy(&mf_weights[0],
					md_synapses.get() + FCM_WEIGHT * w_planeSize,
					w_planeSize, cudaMemcpyDeviceToHost));
	}

	assert(sizeof(weight_t) == sizeof(synapse_t));

	*weight = (weight_t*) &mf_weights[sourceNeuron * wpitch()];
	*partition = &mf_targetPartition[sourceNeuron][0];
	*neuron = &mf_targetNeuron[sourceNeuron][0];
	*plastic = &mf_plastic[sourceNeuron][0];

	assert(mf_targetPartition[sourceNeuron].size() <= wpitch());

	return mf_targetPartition[sourceNeuron].size();
}



size_t
SynapseGroup::planeSize() const
{
	return MAX_PARTITION_SIZE * md_bpitch;
}



size_t
SynapseGroup::dataSize() const
{
	return FCM_SUBMATRICES * planeSize();
}



size_t
SynapseGroup::bpitch() const
{
	return md_bpitch;
}



size_t
SynapseGroup::wpitch() const
{
	return md_bpitch / sizeof(synapse_t);
}
