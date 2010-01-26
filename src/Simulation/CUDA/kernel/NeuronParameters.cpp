#include "NeuronParameters.hpp"

#include <stdexcept>
#include <vector>

#include "kernel.cu_h"
#include "except.hpp"
#include "ThalamicInput.hpp"
#include "nemo_cuda_types.h"


NeuronParameters::NeuronParameters(size_t partitionSize) :
	m_partitionSize(partitionSize)
{ }


void
NeuronParameters::addNeuron(
		nidx_t neuronIndex,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	if(m_acc.find(neuronIndex) != m_acc.end()) {
		//! \todo construct a sensible error message here using sstream
		throw std::runtime_error("duplicate neuron index");
	}
	m_acc[neuronIndex] = nemo::Neuron<float>(a, b, c, d, u, v, sigma);
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
	size_t wpitch = bpitch / sizeof(float);
	md_arr = boost::shared_ptr<float>(d_arr, cudaFree);

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	CUDA_SAFE_CALL(cudaMemset2D(d_arr, bpitch, 0x0, bpitch, height));

	// create host buffer
	std::vector<float> h_arr(height * wpitch, 0);

	// copy data from m_acc to buffer
	size_t planeSize = pcount * wpitch;
	for(acc_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		/*! \todo need to make sure that we use the same mapping here and in
		 * FCM construction. Perhaps wrap this whole thing in a (very simple)
		 * mapper class */
		nidx_t n_idx = i->first % m_partitionSize;
		pidx_t p_idx = i->first / m_partitionSize;
		size_t addr = p_idx * wpitch + n_idx; // address within a plane

		const neuron_t& n = i->second;

		h_arr[PARAM_A * planeSize + addr] = n.a;
		h_arr[PARAM_B * planeSize + addr] = n.b;
		h_arr[PARAM_C * planeSize + addr] = n.c;
		h_arr[PARAM_D * planeSize + addr] = n.d;
		h_arr[STATE_U * planeSize + addr] = n.u;
		h_arr[STATE_V * planeSize + addr] = n.v;
	}

	// copy data across
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, &h_arr[0], height * bpitch, cudaMemcpyHostToDevice));
}
