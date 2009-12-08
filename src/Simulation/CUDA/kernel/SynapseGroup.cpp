//! \file SynapseGroup.cpp

#include "SynapseGroup.hpp"
#include "connectivityMatrix.cu_h"
#include "util.h"


SynapseGroup::SynapseGroup() :
	md_bpitch(0),
	m_partitionSize(0),
	m_allocated(0)
{ }



void
SynapseGroup::addSynapse(
		nidx_t sourceNeuron,
		pidx_t partition,
		nidx_t neuron,
		float weight)
{
	Row& row = mh_synapses[sourceNeuron];
	row.addresses.push_back(f_packSynapse(partition, neuron));
	row.weights.push_back(weight);
	assert(row.addresses.size() == row.weights.size());
}



//! \todo move Synapse type out of this class and pass in just a vector
void
SynapseGroup::addSynapses(
		nidx_t sourceNeuron,
		size_t ncount,
		const pidx_t partition[],
		const nidx_t neuron[],
		const float weight[])
{
	Row& row = mh_synapses[sourceNeuron];
	for(size_t n = 0; n < ncount; ++n) {
		row.addresses.push_back(f_packSynapse(partition[n], neuron[n]));
		row.weights.push_back(weight[n]);
	}
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
SynapseGroup::moveToDevice(size_t partitionSize)
{
	m_partitionSize = partitionSize;

	size_t desiredPitch = maxSynapsesPerNeuron() * sizeof(synapse_t);
	size_t height = FCM_SUBMATRICES * partitionSize;

	synapse_t* d_data = NULL;

	CUDA_SAFE_CALL(
			cudaMallocPitch((void**) &d_data,
				&md_bpitch,
				desiredPitch,
				height));
	m_allocated = md_bpitch * height;

	/* There's no need to clear to allocated memory, as we completely fill it
	 * below */

	size_t wordPitch = md_bpitch / sizeof(synapse_t);
	std::vector<synapse_t> h_data(wordPitch * height, 0); 

	std::vector<synapse_t>::iterator astart = 
			h_data.begin() + FCM_ADDRESS * partitionSize * wordPitch;
	std::vector<synapse_t>::iterator wstart =
			h_data.begin() + FCM_WEIGHT * partitionSize * wordPitch;

	for(std::map<nidx_t, Row>::const_iterator r = mh_synapses.begin();
			r != mh_synapses.end(); ++r) {

		nidx_t sourceNeuron = r->first;
		const Row row = r->second;
		size_t row_idx = wordPitch * sourceNeuron;

		std::copy(row.addresses.begin(), row.addresses.end(), astart + row_idx);
		std::copy(row.weights.begin(), row.weights.end(), wstart + row_idx);
	}

	CUDA_SAFE_CALL(cudaMemcpy(d_data, &h_data[0], m_allocated, cudaMemcpyHostToDevice));
	
	mh_synapses.clear();
	md_synapses = boost::shared_ptr<synapse_t>(d_data, cudaFree);	
	return md_synapses;
}



size_t
SynapseGroup::planeSize() const
{
	return m_partitionSize * md_bpitch;
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
	return md_bpitch * sizeof(synapse_t);
}
