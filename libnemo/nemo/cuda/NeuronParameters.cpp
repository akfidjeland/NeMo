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

#include <nemo/network/Generator.hpp>

#include "types.h"
#include "kernel.cu_h"
#include "device_memory.hpp"
#include "exception.hpp"
#include "kernel.hpp"
#include "Mapper.hpp"


namespace nemo {
	namespace cuda {


NeuronParameters::NeuronParameters(const network::Generator& net, Mapper& mapper) :
	m_allocated(0),
	m_wpitch(0),
	m_pcount(mapper.partitionCount())
{
	size_t height = allocateDeviceData(m_pcount, mapper.partitionSize());

	std::vector<float> h_arr(height * m_wpitch, 0);
	std::map<pidx_t, nidx_t> maxPartitionNeuron;

	size_t veclen = m_pcount * m_wpitch;

	for(network::neuron_iterator i = net.neuron_begin(), i_end = net.neuron_end();
			i != i_end; ++i) {

		DeviceIdx dev = mapper.addIdx(i->first);
		// address within a plane
		size_t addr = dev.partition * m_wpitch + dev.neuron;

		const nemo::Neuron<float>& n = i->second;

		h_arr.at(PARAM_A * veclen + addr) = n.a;
		h_arr.at(PARAM_B * veclen + addr) = n.b;
		h_arr.at(PARAM_C * veclen + addr) = n.c;
		h_arr.at(PARAM_D * veclen + addr) = n.d;
		h_arr.at((NEURON_PARAM_COUNT + STATE_U) * veclen + addr) = n.u;
		h_arr.at((NEURON_PARAM_COUNT + STATE_V) * veclen + addr) = n.v;

		maxPartitionNeuron[dev.partition] =
			std::max(maxPartitionNeuron[dev.partition], dev.neuron);
	}

	memcpyToDevice(md_arr.get(), h_arr, height * m_wpitch);
	configurePartitionSizes(maxPartitionNeuron);
}



size_t
NeuronParameters::allocateDeviceData(size_t pcount, size_t psize)
{
	size_t width = psize * sizeof(float);
	size_t height = (NEURON_PARAM_COUNT + NEURON_STATE_COUNT) * pcount;
	size_t bpitch = 0;

	float* d_arr;
	d_mallocPitch((void**)&d_arr, &bpitch, width, height, "neuron parameters");
	m_wpitch = bpitch / sizeof(float);
	md_arr = boost::shared_ptr<float>(d_arr, d_free);
	m_allocated = height * bpitch;

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	d_memset2D(d_arr, bpitch, 0x0, height);

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


float*
NeuronParameters::d_parameters() const
{
	//! \todo remove hard-coded assumptions regarding parameter/state order
	return md_arr.get();
}


float*
NeuronParameters::d_state() const
{
	//! \todo remove hard-coded assumptions regarding parameter/state order
	return md_arr.get() + NEURON_PARAM_COUNT * m_pcount * wordPitch();
}


	} // end namespace cuda
} // end namespace nemo
