//! \file DeviceMemory.cpp

#include "DeviceMemory.hpp"
#include "Cluster.hpp"
#include "izhikevich_kernel.h"

#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>



DeviceMemory::DeviceMemory(std::vector<Cluster>& clusters, 
		bool forceDense,
		bool verbose) :
	pitch32(0),
	pitch8(0),
	pitch1(0),
	m_clusterCount(clusters.size()),
	m_maxDelay(1),
	m_sparseEncoding(false),
	m_verbose(verbose),
	m_hasExternalCurrent(MAX_THREAD_BLOCKS, 0),
	m_hasExternalFiring(MAX_THREAD_BLOCKS, 0),
	m_maxColumnIndex(MAX_THREAD_BLOCKS, 0)
{
	int maxClusterSize = (max_element(clusters.begin(), clusters.end()))->n;
	int minClusterSize = (min_element(clusters.begin(), clusters.end()))->n;
	
	for(std::vector<Cluster>::const_iterator c=clusters.begin();
			c!=clusters.end(); ++c) {
		m_maxDelay = std::max(m_maxDelay, c->maxDelay());
	}

	//! \todo deal with clusters of differing size
	assert(maxClusterSize == minClusterSize);

	/*! \todo the current allocation is wasteful if we use sparse encoding.
	 * Instead of allocating the required space for a dense encoding, allocate
	 * according to the maximum pitch between *all* clusters */
	allocateData(m_clusterCount, maxClusterSize);

	for(std::vector<Cluster>::const_iterator c=clusters.begin();
			c!=clusters.end(); ++c) {
		copyData(c-clusters.begin(), *c, forceDense);
	}

	setFiringDelays(clusters);

	clearBuffers(m_clusterCount);

	if(verbose) {
		std::cerr << "Sparse encoding: " << m_sparseEncoding << "\n";
		std::cerr << "32b pitch: " << pitch32 << "\n";
		std::cerr << "8b pitch: " << pitch8 << "\n";
		std::cerr << "1b pitch: " << pitch1 << "\n";
		std::cerr << "Max row length: ";
		std::copy(m_maxColumnIndex.begin(), m_maxColumnIndex.begin() + clusters.size(), 
				std::ostream_iterator<int>(std::cerr, " "));
		std::cerr << std::endl;
	}

	//! \todo report space usage here
}



DeviceMemory::~DeviceMemory()
{
	cudaFree(weights);
	cudaFree(firing);
	cudaFree(firingDelays);
	cudaFree(extI);
	cudaFree(extFiring);
	cudaFree(d);
	cudaFree(c);
	cudaFree(b);
	cudaFree(a);
	cudaFree(u);
	cudaFree(v);
}



//=============================================================================
// Addressing 
//=============================================================================


int*
DeviceMemory::firingAddress(int clusterIndex)
{
	if(clusterIndex < m_clusterCount) {
		return firing + clusterIndex*pitch32/4;
	} else {
		return NULL;
	}
}




float*
DeviceMemory::vAddress(int clusterIndex)
{
	if(clusterIndex < m_clusterCount) {
		return (float*)((char*)v + clusterIndex*pitch32);
	} else {
		return NULL;
	}
}



//=============================================================================
// Allocation/copy
//=============================================================================



/*! Allocate memory and check that pitch is as expected */
void
alloc2D(void **arr, size_t* pitch, int width, int height)
{
	size_t newPitch;
	cudaError_t status;

	if((status = cudaMallocPitch(arr, &newPitch, width, height)) != cudaSuccess) {
		std::stringstream msg; 
		msg << "Error: failed to allocate memory (errno:" << status << ")\n";
		throw cuda_mem_error(msg.str());
	}

	if(*pitch != 0 && *pitch != newPitch) {
		std::stringstream msg;
		msg << "Pitch mismatch in device memory allocation ("
			<< pitch << "!=" << newPitch << ")\n";
		throw cuda_mem_error(msg.str());
	} else {
		*pitch = newPitch;
	}

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	if((status = cudaMemset2D(*arr, newPitch, 0x0, newPitch, height)) != cudaSuccess) {
		std::stringstream msg;
		msg << "Error: failed to clear array (errno:" << status << ")\n";
		throw cuda_mem_error(msg.str());
	}
}



/* Allocate all data as 2D array to ensure pitch (between clusters) avoids bank
 * conflicts */ 
void
DeviceMemory::allocateData(int clusterCount, int clusterSize)
{
	//! \todo report space usage *before* allocating, in case of error
	
	alloc2D((void**)&v, &pitch32, clusterSize*sizeof(float), clusterCount);
	alloc2D((void**)&u, &pitch32, clusterSize*sizeof(float), clusterCount);
	alloc2D((void**)&a, &pitch32, clusterSize*sizeof(float), clusterCount);
	alloc2D((void**)&b, &pitch32, clusterSize*sizeof(float), clusterCount);
	alloc2D((void**)&c, &pitch32, clusterSize*sizeof(float), clusterCount);
	alloc2D((void**)&d, &pitch32, clusterSize*sizeof(float), clusterCount);

	alloc2D((void**)&extI, &pitch32, clusterSize*sizeof(float), clusterCount);
	alloc2D((void**)&extFiring, &pitch1, clusterSize*sizeof(uint32_t)/32, clusterCount);

	/* Array of densely packed spikes, for up to the 32 most recent simulation cycles */
	alloc2D((void**)&firing, &pitch32, clusterSize*sizeof(uint32_t), clusterCount);
	alloc2D((void**)&firingDelays, &pitch32, clusterSize*sizeof(uint32_t), clusterCount);

	/* The whole collection of connectivity matrices are now a 3D structure.
	 * However, we can write neighbouring matrices back-to-back and get
	 * alligned data for each row. */
	alloc2D((void**)&weights, &pitch32,
			clusterSize*sizeof(float),
			clusterSize*clusterCount);

#ifndef BIT_PACK_DELAYS
	alloc2D((void**)&delays, &pitch8,
			clusterSize,
			clusterSize*clusterCount);
#endif
}



template<typename T>
void
copyArr(T* dst,
		const T* src,
		size_t row,
		size_t pitch,
		size_t rowLength)
{
	assert(rowLength*sizeof(T) <= pitch );
	//! \todo detect copy errors
	cudaMemcpy((char*)dst+row*pitch, src, rowLength*sizeof(T), cudaMemcpyHostToDevice);
}



inline
int 
log2ceil(unsigned int n)
{
	unsigned int v = n;
	unsigned int r = 0; 
	while (v >>= 1) 
		r++;
	return n == unsigned(1)<<r ? r : r+1;
}



bool
DeviceMemory::denseEncoding(const Cluster& cluster) const
{
	/*! \todo If we modify the sparse encoding to avoid bank conflicts this
	 * pitch is no longer valid. */
	unsigned int minRowPitch = cluster.maxRowEntries()*sizeof(int2);
	/* It would be possible to use sparse encoding up to pitch32, rather than
	 * pitch32/2, but we use this limit as the dense encoding avoids some
	 * unpacking in the kernel */ 
	return minRowPitch >= pitch32/2;
}



/*! Write sparse connectivity matrix in the form postsynaptic index (2 bytes),
 * delay (1 byte), padding (1 bytes), strength (4 bytes). 
 *
 * If care is not taken in writing the dense matrix, the kernel might end up
 * with shared memory bank conflicts. The reason is that, unlike in dense mode,
 * thread indices can point to arbitrary postsynaptic neurons. It is thus
 * possible to have several threads within a warp accessing a postsynaptic
 * neuron in the same bank. 
 *
 * \todo cater for this
 */
void
DeviceMemory::copySparseConnectivity(int clusterIndex, const Cluster& cluster)
{
	m_sparseEncoding = true;
	int maxColumnIndex = 0; 

	for(int pre=0; pre < cluster.n; ++pre){
		std::vector<int2> row(pitch32/sizeof(int2), make_int2(0,0));
		std::vector<int2>::iterator next = row.begin();
		for(int post=0; post < cluster.n; ++post){
			unsigned char delay = cluster.connectionDelay(pre, post);
			float strength = cluster.connectionStrength(pre, post);
			short nn = short(post);
			if(strength != 0.0f) {
				if(next >= row.end()) {
					throw std::out_of_range("DeviceMemory::copySparseConnectivity: beyond end of array");
				}
				next->x = (nn << 16) | delay;
				next->y = reinterpret_cast<int&>(strength);
				++next;
				maxColumnIndex = std::max(maxColumnIndex, next-row.begin());
			}
		}
		cudaMemcpy((char*)weights + clusterIndex*cluster.n*pitch32 + pre*pitch32,
				&row[0],
				pitch32,
				cudaMemcpyHostToDevice);
	}
	m_maxColumnIndex[clusterIndex] = maxColumnIndex;
}



void
DeviceMemory::copyDenseConnectivity(int clusterIndex, const Cluster& cluster)
{
	for( int i=0; i < cluster.n; ++i ){
		std::vector<uint32_t> bits(cluster.n, 0);
		memcpy(&bits[0], cluster.connectionStrength()+i*cluster.n,
				cluster.n*sizeof(float));
#ifdef BIT_PACK_DELAYS
		for( int j=0; j < cluster.n; ++j ){
			unsigned char dbits = cluster.connectionDelay(i,j);
			//! \todo check that high-order bits are not set
			dbits &= 0x1f;
			//! \todo careful with byte-order here!
			uint32_t wbits = bits[j] & ~0x1f;
			wbits |= dbits;
			bits[j] = wbits;
		}
#endif
		cudaMemcpy((char*)weights + clusterIndex*cluster.n*pitch32 + i*pitch32,
				&bits[0],
				cluster.n*sizeof(float),
				cudaMemcpyHostToDevice);
	}

#ifndef BIT_PACK_DELAYS
	for( int i=0; i < cluster.n; ++i ){
		cudaMemcpy((char*)delays + clusterIndex*cluster.n*pitch8 +  i*pitch8,
				cluster.connectionDelay()+i*cluster.n, 
				cluster.n,
				cudaMemcpyHostToDevice);
	}
#endif
	m_maxColumnIndex[clusterIndex] = DENSE_ENCODING;
}



void
DeviceMemory::copyData(int clusterIndex, const Cluster& cluster, bool forceDense)
{
	//! \todo fix this!
	n = cluster.n;

	copyArr(v, cluster.v(), clusterIndex, pitch32, cluster.n);
	copyArr(u, cluster.u(), clusterIndex, pitch32, cluster.n);

	copyArr(a, cluster.a(), clusterIndex, pitch32, cluster.n);
	copyArr(b, cluster.b(), clusterIndex, pitch32, cluster.n);
	copyArr(c, cluster.c(), clusterIndex, pitch32, cluster.n);
	copyArr(d, cluster.d(), clusterIndex, pitch32, cluster.n);

	if(forceDense || denseEncoding(cluster)) {
		copyDenseConnectivity(clusterIndex, cluster);
	} else {
		copySparseConnectivity(clusterIndex, cluster);
	}

	// set configuration flags
	m_hasExternalCurrent[clusterIndex] = cluster.hasExternalCurrent() ? 1 : 0;
	m_hasExternalFiring[clusterIndex] = cluster.hasExternalFiring() ? 1 : 0;
}



/*! Set the firing delays bit-fields for each cluster and copy data to device */
void
DeviceMemory::setFiringDelays(const std::vector<Cluster>& clusters)
{
	int delayAcc = 0;
	int delayCount = 0;

	size_t size = clusters.size()*pitch32;
	std::vector<uint32_t> delayArr(size/sizeof(uint32_t), 0);

	for(std::vector<Cluster>::const_iterator c=clusters.begin();
			c!=clusters.end(); ++c) {

		int clusterIndex = c-clusters.begin();

		for(int pre=0; pre < c->n; ++pre){
			uint32_t delayBits = 0;
			for(int post=0; post < c->n; ++post){
				unsigned char delay = c->connectionDelay(pre, post);
				if(delay) {
					delayBits |= 0x1 << (delay-1);	
				}
			}
			delayArr[clusterIndex*pitch32/sizeof(uint32_t) + pre] = delayBits;

			// count number of unique delays
			int c;
			int v = delayBits;
			for(c=0; v; v >>= 1)
				c += v & 1;
			delayAcc += c;
			delayCount += 1;
		}
	}

	if(m_verbose) {
		std::cerr << "Average delay bits set: " << delayAcc/delayCount << std::endl;
	}

	cudaMemcpy((char*)firingDelays, &delayArr[0], size, cudaMemcpyHostToDevice);
}



void
DeviceMemory::clearBuffers(int clusterCount)
{
	/* Not really needed for initial current since we always set this */
	cudaMemset(extI, 0, clusterCount*pitch32);
	//! \todo set only the required size here!
	cudaMemset(extFiring, 0, clusterCount*pitch1);
	cudaMemset(firing, 0, clusterCount*pitch32);
}



//=============================================================================
// Configuration
//=============================================================================


const char*
DeviceMemory::hasExternalCurrent() const
{
	return &m_hasExternalCurrent[0];
}



const char*
DeviceMemory::hasExternalFiring() const
{
	return &m_hasExternalFiring[0];
}



const int* 
DeviceMemory::maxColumnIndex() const
{
	return &m_maxColumnIndex[0];
}



unsigned char
DeviceMemory::maxDelay() const
{
	return m_maxDelay;
}
