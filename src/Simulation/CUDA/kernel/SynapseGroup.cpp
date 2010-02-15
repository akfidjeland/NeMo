//! \file SynapseGroup.cpp

#include <cmath>
#include <sstream>

#include "SynapseGroup.hpp"

#include "except.hpp"
#include "connectivityMatrix.cu_h"
#include "util.h"
#include "kernel.cu_h"
#include "fixedpoint.hpp"


SynapseGroup::SynapseGroup() :
	m_lastSync(-1),
	m_maxAbsWeight(0.0f)
{ }



sidx_t
SynapseGroup::addSynapse(
		nidx_t sourceNeuron,
		pidx_t partition,
		nidx_t neuron,
		float weight,
		uchar plastic)
{
	row_t& row = mh_synapses[sourceNeuron];
	row.push_back(boost::tuple<nidx_t, weight_t>(neuron, weight));

	//! \todo only construct this once we have reordered data
	mf_targetPartition[sourceNeuron].push_back(partition);
	mf_targetNeuron[sourceNeuron].push_back(neuron);
	mf_plastic[sourceNeuron].push_back(plastic);

	m_maxAbsWeight = std::max(fabsf(weight), m_maxAbsWeight);

	return row.size() - 1;
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
	row_t& row = mh_synapses[sourceNeuron];
	for(size_t n = 0; n < ncount; ++n) {
		row.push_back(boost::tuple<nidx_t, weight_t>(neuron[n], weight[n]));
		m_maxAbsWeight = std::max(fabsf(weight[n]), m_maxAbsWeight);
	}

	std::copy(partition, partition+ncount, back_inserter(mf_targetPartition[sourceNeuron]));
	std::copy(neuron, neuron+ncount, back_inserter(mf_targetNeuron[sourceNeuron]));
	std::copy(plastic, plastic+ncount, back_inserter(mf_plastic[sourceNeuron]));
}




/*! fill host buffer with synapse data */
size_t
SynapseGroup::fillFcm(
		uint fractionalBits,
		size_t startWarp,
		size_t totalWarps,
		std::vector<synapse_t>& h_data)
{
	size_t writtenWarps = 0; // warps

	std::vector<synapse_t> addresses;
	std::vector<weight_dt> weights;

	for(std::map<nidx_t, row_t>::const_iterator r = mh_synapses.begin();
			r != mh_synapses.end(); ++r) {

		nidx_t sourceNeuron = r->first;
		const row_t& row = r->second;

		m_warpOffset[sourceNeuron] = startWarp + writtenWarps;

		addresses.resize(row.size());
		weights.resize(row.size());
		for(size_t sidx = 0; sidx < row.size(); ++sidx) {

			const h_synapse_t& s = row.at(sidx);
			nidx_t neuron = boost::tuples::get<0>(s);
			addresses.at(sidx) = f_packSynapse(boost::tuples::get<0>(s));
			weights.at(sidx) = fixedPoint(boost::tuples::get<1>(s), fractionalBits);

			uint twarp = neuron / WARP_SIZE;
			uint gwarp = startWarp + writtenWarps + sidx / WARP_SIZE;
			//! \todo add assertions here
			//! \todo do a double loop here to save map lookups
			uint32_t warpBit = 0x1 << twarp;
			m_warpTargets[gwarp] |= warpBit;
		}

		assert(sizeof(nidx_t) == sizeof(synapse_t));
		assert(sizeof(weight_dt) == sizeof(synapse_t));

		synapse_t* aptr = &h_data.at((startWarp + writtenWarps) * WARP_SIZE);
		synapse_t* wptr = &h_data.at((totalWarps + startWarp + writtenWarps) * WARP_SIZE);

		/*! note that std::copy won't work as it will silently cast floats to integers */
		memcpy(aptr, &addresses[0], addresses.size() * sizeof(synapse_t));
		memcpy(wptr, &weights[0], weights.size() * sizeof(synapse_t));

		writtenWarps += DIV_CEIL(row.size(), WARP_SIZE);
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



//! \todo merge with warpOffset. Just return a pair here
uint32_t
SynapseGroup::warpTargetBits(nidx_t neuron, size_t warp) const
{
	uint32_t gwarp = warpOffset(neuron, warp);
	std::map<uint, uint32_t>::const_iterator entry = m_warpTargets.find(gwarp);
	if(m_warpTargets.end() == entry) {
		throw std::runtime_error("warp not found");
	}
	return entry->second;
}

