#ifndef NEMO_CUDA_NVECTOR_CU
#define NEMO_CUDA_NVECTOR_CU

/*! \file nvector.cu Access functions for per-neuron data.
 *
 * See NVector.hpp/NVector.ipp for host-side functionality
 */

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

__constant__ size_t c_pitch32;
__constant__ size_t c_pitch64;


__host__
cudaError
nv_setPitch32(size_t pitch32)
{
	return cudaMemcpyToSymbol(c_pitch32, &pitch32, sizeof(size_t), 0, cudaMemcpyHostToDevice);
}


__host__
cudaError
nv_setPitch64(size_t pitch64)
{
	return cudaMemcpyToSymbol(c_pitch64, &pitch64, sizeof(size_t), 0, cudaMemcpyHostToDevice);
}



/*! \return 32-bit datum for a single neuron in the current partition */
template<typename T>
__device__
T
nv_load32(unsigned neuron, unsigned plane, T* g_data)
{
	return g_data[(plane * PARTITION_COUNT + CURRENT_PARTITION) * c_pitch32 + neuron];
}


/*! \return 64-bit datum for a single neuron in the current partition */
template<typename T>
__device__
T
nv_load64(unsigned neuron, unsigned plane, T* g_data)
{
	return g_data[(plane * PARTITION_COUNT + CURRENT_PARTITION) * c_pitch64 + neuron];
}

#endif
