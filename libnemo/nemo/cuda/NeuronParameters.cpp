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
#include <vector>

#include "Mapper.hpp"
#include "kernel.cu_h"
#include "except.hpp"
#include "NetworkImpl.hpp"
#include "cuda_types.h"
#include "kernel.hpp"


namespace nemo {
	namespace cuda {


NeuronParameters::NeuronParameters(const nemo::NetworkImpl& net, const Mapper& mapper) :
	m_allocated(0),
	m_wpitch(0)
{
	size_t height = allocateDeviceData(mapper.partitionCount(), mapper.partitionSize());

	std::vector<float> h_arr(height * m_wpitch, 0);
	std::map<pidx_t, nidx_t> maxPartitionNeuron;

	size_t veclen = mapper.partitionCount() * m_wpitch;

	for(std::map<nidx_t, nemo::Neuron<float> >::const_iterator i = net.m_neurons.begin();
			i != net.m_neurons.end(); ++i) {

		DeviceIdx dev = mapper.deviceIdx(i->first);
		// address within a plane
		size_t addr = dev.partition * m_wpitch + dev.neuron;

		const nemo::Neuron<float>& n = i->second;

		h_arr.at(PARAM_A * veclen + addr) = n.a;
		h_arr.at(PARAM_B * veclen + addr) = n.b;
		h_arr.at(PARAM_C * veclen + addr) = n.c;
		h_arr.at(PARAM_D * veclen + addr) = n.d;
		h_arr.at(STATE_U * veclen + addr) = n.u;
		h_arr.at(STATE_V * veclen + addr) = n.v;

		maxPartitionNeuron[dev.partition] =
			std::max(maxPartitionNeuron[dev.partition], dev.neuron);
	}

	// copy data to device
	size_t bpitch = m_wpitch * sizeof(float);
	CUDA_SAFE_CALL(cudaMemcpy(md_arr.get(), &h_arr[0], height * bpitch, cudaMemcpyHostToDevice));
	configurePartitionSizes(maxPartitionNeuron);
}



size_t
NeuronParameters::allocateDeviceData(size_t pcount, size_t psize)
{
	size_t width = psize * sizeof(float);
	size_t height = NVEC_COUNT * pcount;
	size_t bpitch = 0;

	float* d_arr;
	cudaError err = cudaMallocPitch((void**)&d_arr, &bpitch, width, height);
	if(cudaSuccess != err) {
		throw DeviceAllocationException("neuron parameters", width * height, err);
	}
	m_wpitch = bpitch / sizeof(float);
	md_arr = boost::shared_ptr<float>(d_arr, cudaFree);
	m_allocated = height * bpitch;

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	CUDA_SAFE_CALL(cudaMemset2D(d_arr, bpitch, 0x0, bpitch, height));

	return height;
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

	CUDA_SAFE_CALL(configurePartitionSize(&partitionSizes[0], partitionSizes.size()));
}

	} // end namespace cuda
} // end namespace nemo
