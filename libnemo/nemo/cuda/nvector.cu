/* Access functions for per-neuron data.
 *
 * See NVector.hpp/NVector.ipp for host-side functionality
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





