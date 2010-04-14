/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "NeuronParameters.hpp"

#include <stdexcept>
#include <vector>

#include "kernel.cu_h"
#include "except.hpp"
#include "Network.hpp"
#include "ThalamicInput.hpp"
#include "cuda_types.h"
#include "kernel.hpp"

namespace nemo {
	namespace cuda {


NeuronParameters::NeuronParameters(const nemo::Network& net, size_t partitionSize) :
	m_partitionSize(partitionSize),
	m_allocated(0),
	m_wpitch(0)
{
	//! \todo change to using internal data in nemo::Network directly
	for(std::map<nidx_t, nemo::Neuron<float> >::const_iterator i = net.m_neurons.begin();
			i != net.m_neurons.end(); ++i) {
		addNeuron(i->first, i->second);
	}
	moveToDevice();
}



void
NeuronParameters::addNeuron(nidx_t nidx, const nemo::Neuron<float>& n)
{
	if(m_acc.find(nidx) != m_acc.end()) {
		//! \todo construct a sensible error message here using sstream
		throw std::runtime_error("duplicate neuron index");
	}
	m_acc[nidx] = n;

	//! \todo share mapper code with moveToDevice and ConnectivityMatrixImpl
	nidx_t ni = nidx % m_partitionSize;
	pidx_t pi = nidx / m_partitionSize;

	m_maxPartitionNeuron[pi] = std::max(m_maxPartitionNeuron[pi], ni);
}



void
NeuronParameters::setSigma(ThalamicInput& th) const
{
	for(acc_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		//! \todo share mapper code with moveToDevice and ConnectivityMatrixImpl
		nidx_t n = i->first % m_partitionSize;
		pidx_t p = i->first / m_partitionSize;
		th.setNeuronSigma(p, n, i->second.sigma);
	}	
}



nidx_t
NeuronParameters::maxNeuronIdx() const
{
	if(m_acc.size() == 0) {
		return 0;
	} else {
		// map is guaranteed to be sorted
		return m_acc.rbegin()->first;
	}
}

size_t
NeuronParameters::partitionCount() const
{
	nidx_t max = maxNeuronIdx();
	return (0 == max) ? 0 : DIV_CEIL(max+1, m_partitionSize);
}


void
NeuronParameters::moveToDevice()
{
	//! \todo could just allocate sigma here as well
	const size_t planeCount = 6; // a-d, u, v
	const size_t pcount = partitionCount();

	size_t width = m_partitionSize * sizeof(float);
	size_t height = planeCount * pcount;
	size_t bpitch = 0;

	//! make this possibly failable
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

	// create host buffer
	std::vector<float> h_arr(height * m_wpitch, 0);

	// copy data from m_acc to buffer
	for(acc_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		/*! \todo need to make sure that we use the same mapping here and in
		 * FCM construction. Perhaps wrap this whole thing in a (very simple)
		 * mapper class */
		nidx_t n_idx = i->first % m_partitionSize;
		pidx_t p_idx = i->first / m_partitionSize;
		size_t addr = p_idx * m_wpitch + n_idx; // address within a plane

		const neuron_t& n = i->second;

		h_arr[PARAM_A * veclen + addr] = n.a;
		h_arr[PARAM_B * veclen + addr] = n.b;
		h_arr[PARAM_C * veclen + addr] = n.c;
		h_arr[PARAM_D * veclen + addr] = n.d;
		h_arr[STATE_U * veclen + addr] = n.u;
		h_arr[STATE_V * veclen + addr] = n.v;
	}

	// copy data across
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, &h_arr[0], height * bpitch, cudaMemcpyHostToDevice));

	configurePartitionSizes();
}



void
NeuronParameters::configurePartitionSizes()
{
	if(m_maxPartitionNeuron.size() == 0) {
		return;
	}

	size_t maxPidx = m_maxPartitionNeuron.rbegin()->first;
	std::vector<uint> partitionSizes(maxPidx, 0);

	for(std::map<pidx_t, nidx_t>::const_iterator i = m_maxPartitionNeuron.begin();
			i != m_maxPartitionNeuron.end(); ++i) {
		partitionSizes[i->first] = i->second + 1;
	}

	configurePartitionSize(&partitionSizes[0], partitionSizes.size());
}

	} // end namespace cuda
} // end namespace nemo
