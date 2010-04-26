/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "NeuronParameters.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

#include "DeviceIdx.hpp"
#include "kernel.cu_h"
#include "except.hpp"
#include "Network.hpp"
#include "cuda_types.h"
#include "kernel.hpp"
#include "util.h"


namespace nemo {
	namespace cuda {


NeuronParameters::NeuronParameters(
		const nemo::Network& net,
		const Mapper& mapper,
		size_t partitionSize) :
	m_allocated(0),
	m_wpitch(0)
{
	acc_t acc; // accumulor for neuron data
	std::map<pidx_t, nidx_t> maxPartitionNeuron;
	addNeurons(net, mapper, &acc, &maxPartitionNeuron);
	// map is guaranteed to be sorted
	nidx_t maxIdx = acc.size() == 0 ? 0 : acc.rbegin()->first;
	m_partitionCount = (0 == maxIdx) ? 0 : DIV_CEIL(maxIdx+1, partitionSize);
	moveToDevice(acc, mapper, partitionSize);
	configurePartitionSizes(maxPartitionNeuron);
}



void
NeuronParameters::addNeurons(
		const nemo::Network& net,
		const Mapper& mapper,
		acc_t* acc,
		std::map<pidx_t, nidx_t>* maxPartitionNeuron)
{
	for(std::map<nidx_t, nemo::Neuron<float> >::const_iterator i = net.m_neurons.begin();
			i != net.m_neurons.end(); ++i) {

		nidx_t nidx = i->first;
		const nemo::Neuron<float>& n = i->second;

		std::pair<acc_t::iterator, bool> insertion =
			acc->insert(std::make_pair(nidx, n));
		if(!insertion.second) {
			std::ostringstream msg;
			msg << "Multiple neurons specified for neuron index " << nidx;
			throw std::runtime_error(msg.str());
		}

		DeviceIdx dev = mapper.deviceIdx(nidx);

		(*maxPartitionNeuron)[dev.partition] =
			std::max((*maxPartitionNeuron)[dev.partition], dev.neuron);
	}
}



void
NeuronParameters::moveToDevice(const acc_t& acc,
		const Mapper& mapper, size_t partitionSize)
{
	//! \todo could just allocate sigma here as well
	const size_t pcount = m_partitionCount;

	size_t width = partitionSize * sizeof(float);
	size_t height = NVEC_COUNT * pcount;
	size_t bpitch = 0;

	float* d_arr;
	cudaError err = cudaMallocPitch((void**)&d_arr, &bpitch, width, height);
	if(cudaSuccess != err) {
		throw DeviceAllocationException("neuron parameters", width * height, err);
	}
	m_wpitch = bpitch / sizeof(float);
	size_t veclen = pcount * m_wpitch;
	md_arr = boost::shared_ptr<float>(d_arr, cudaFree);
	m_allocated = height * bpitch;

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	CUDA_SAFE_CALL(cudaMemset2D(d_arr, bpitch, 0x0, bpitch, height));

	//! \todo write data directly to buffer. No need for intermediate map structure
	// create host buffer
	std::vector<float> h_arr(height * m_wpitch, 0);

	// copy data from accumulator to buffer
	for(acc_t::const_iterator i = acc.begin(); i != acc.end(); ++i) {
		DeviceIdx dev = mapper.deviceIdx(i->first);
		// address within a plane
		size_t addr = dev.partition * m_wpitch + dev.neuron;

		const neuron_t& n = i->second;

		h_arr.at(PARAM_A * veclen + addr) = n.a;
		h_arr.at(PARAM_B * veclen + addr) = n.b;
		h_arr.at(PARAM_C * veclen + addr) = n.c;
		h_arr.at(PARAM_D * veclen + addr) = n.d;
		h_arr.at(STATE_U * veclen + addr) = n.u;
		h_arr.at(STATE_V * veclen + addr) = n.v;
	}

	// copy data across
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, &h_arr[0], height * bpitch, cudaMemcpyHostToDevice));
}



void
NeuronParameters::configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron)
{
	if(maxPartitionNeuron.size() == 0) {
		return;
	}

	size_t maxPidx = maxPartitionNeuron.rbegin()->first;
	std::vector<unsigned> partitionSizes(maxPidx+1, 0);

	for(std::map<pidx_t, nidx_t>::const_iterator i = maxPartitionNeuron.begin();
			i != maxPartitionNeuron.end(); ++i) {
		partitionSizes.at(i->first) = i->second + 1;
	}

	configurePartitionSize(&partitionSizes[0], partitionSizes.size());
}

	} // end namespace cuda
} // end namespace nemo
