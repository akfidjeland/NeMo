//! \file SynapseGroup.cpp

#include "SynapseGroup.hpp"

#include "except.hpp"
#include "connectivityMatrix.cu_h"
#include "util.h"
#include "kernel.cu_h"


SynapseGroup::SynapseGroup() :
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




/*! fill host buffer with synapse data */
size_t
SynapseGroup::fillFcm(size_t startWarp, size_t totalWarps, std::vector<synapse_t>& h_data)
{
	size_t writtenWarps = 0; // warps

	for(std::map<nidx_t, Row>::const_iterator r = mh_synapses.begin();
			r != mh_synapses.end(); ++r) {

		nidx_t sourceNeuron = r->first;
		const Row row = r->second;

		m_warpOffset[sourceNeuron] = startWarp + writtenWarps;

		synapse_t* aptr = &h_data.at((startWarp + writtenWarps) * WARP_SIZE);
		synapse_t* wptr = &h_data.at((totalWarps + startWarp + writtenWarps) * WARP_SIZE);

		/*! note that std::copy won't work as it will silently cast floats to integers */
		memcpy(aptr, &row.addresses[0], row.addresses.size() * sizeof(synapse_t));
		memcpy(wptr, &row.weights[0], row.weights.size() * sizeof(synapse_t));

		assert(row.addresses.size() == row.weights.size());

		writtenWarps += DIV_CEIL(row.addresses.size(), WARP_SIZE);
	}

	return writtenWarps;
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
	//! \todo need to add this back, using the new FCM format
#if 0
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
#endif
	return 0;
}



uint32_t
SynapseGroup::warpOffset(nidx_t neuron, size_t warp) const
{
	std::map<nidx_t, size_t>::const_iterator entry = m_warpOffset.find(neuron);
	if(m_warpOffset.end() == entry) {
		throw std::runtime_error("neuron not found");
	}
	return entry->second + warp;
}

