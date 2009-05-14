//! \file izhikevich_kernel.cu

/* \brief GPU/CUDA kernel for neural simulation using Izhikevich's model
 * 
 * The entry point for the kernel is stepSimulation which will do a single
 * integrate-and-fire step.  
 *
 * \author Andreas Fidjeland
 */

#include "izhikevich.h"
#include "izhikevich_kernel.h"
#include "DeviceMemory.hpp"
#include <assert.h>


//#define THREADS_PER_BLOCK 64
//#define THREADS_PER_BLOCK 128
#define THREADS_PER_BLOCK 256

/* Experimented with different chunk sizes here. There are variations of a
 * few % of execution time. 32 and 16 seems to work best */
/*! \bug If chunkSize == 128, the kernel hangs */
#define FIRING_IDX_CHUNK_SIZE (16+1)

// A job must be at least the size of a warp
//! \todo warps size might change in later generations 
#define MAX_STREAMS (THREADS_PER_BLOCK/32)

#define SLOW_32B_INTS

#ifdef SLOW_32B_INTS
#define mul24(a,b) __mul24(a,b)
#else
#define mul24(a,b) a*b
#endif



/* The amount of external memory required per thread block depends on the
 * number of neurons per block, which is only known at run time */
extern __shared__ char sMem[];


/*! Different clusters may have slightly different configuration, e.g. with
 * respect to what external inputs they require. This information is all kept
 * in constant memory and is thus cached. 
 * 
 * It is not possible to dynamically size data structures in constant memory,
 * so we simply set some upper limit on the number of thread blocks
 * (MAX_THREAD_BLOCKS) and size the data structures statically.
 */


/*! Per-cluster flag indicating whether the cluster receives an external input
 * current from the host */
__constant__ char cHasExternalCurrent[MAX_THREAD_BLOCKS]; 
/*! Per-cluster flag indicating whether the cluster receives external input
 * firing, i.e. whether the host can force a neuron to fire */
__constant__ char cHasExternalFiring[MAX_THREAD_BLOCKS];



/*! Per cluster configuration parameter specifygin the maximum length of rows
 * in a sparsely encoded connectivity matrix. If the encoding is dense, this
 * parameter is set to DENSE_ENCODING */
__constant__ int cMaxColumnIndex[MAX_THREAD_BLOCKS];





/* The number of neurons in the cluster may not be an exact multiple of the
 * number of threads in a block. Thus, some threads will idle at some times. To
 * avoid wasting bandwidth, we should not do as little external memory accesses
 * as possible. However, we should also take care to avoid warp divergence */

/*
 * It's possible that one warp has some active threads and some inactive. With
 * some care in memory allocation we can avoid divergence within this mixed
 * warp. We have to make sure that the whole warp can read and write to global
 * memory, and must thus have sufficient padding at the end of each row of
 * data.
 * 
 * |- threads/cluster --|
 * |---- neurons/cluster ----|
 * |---- active threads -----|
 * |--------- active warps --------|                
 * |--- w ---||--- w ---||--- w ---||---- w ---|    Warps
 * |----- neuron 0 -----||---- neuron 1 -------|    Neuron index within thread
 * 
 * 
 * 
 * 
 */


/* \param ws warp size
 * \param n  number of data
 * \return number of data rounded up to the nearest warp size
 */
__device__ 
__host__
int
warpAlign(int n, int ws)
{
	return ws * ( n / ws + ( n % ws ? 1 : 0 ) );
}



/*! \param chunk 
 *		Each thread processes n neurons. The chunk is the index (<n) of the
 *		neuron currently processed by the thread.
 * \return 
 *		true if the thread should be active when processing the specified
 * 		neuron, false otherwise.
 */
__device__
bool 
activeNeuron(int chunk, int neuronsPerBlock)
{
	return threadIdx.x + chunk*THREADS_PER_BLOCK < neuronsPerBlock;
}



__device__
bool
activeWarp(int threadNeuronIndex, int neuronsPerBlock)
{
	return threadIdx.x + threadNeuronIndex*THREADS_PER_BLOCK < 
		warpAlign(neuronsPerBlock, warpSize);
}



__device__
int
neuronsPerThread(int neuronsPerBlock)
{
	return neuronsPerBlock / THREADS_PER_BLOCK
	     + (neuronsPerBlock % THREADS_PER_BLOCK ? 1 : 0);
}




//=============================================================================
// Firing buffer
//=============================================================================
// The firing buffer stores for each neuron the spikes on the 32 most recent
// cycles, all packed into a single int
//=============================================================================


/*! \return size (in bytes) of firing buffer */
__device__ __host__
size_t
firingBufferSize(int neuronsPerBlock, size_t warpSize)
{
	return warpAlign(neuronsPerBlock, warpSize)*sizeof(int);
}



/*! \return size (in bytes) of buffer containing firing indices. The group of
 * MAX_STREAMS entries at the beginning is the starting point of each stream.
 * The group of MAX_STREAMS entries past the end contain the length of each
 * stream */
__device__ __host__
size_t
firingIdxBufferSize(int neuronsPerBlock, size_t warpSize)
{
	return warpAlign(MAX_STREAMS+neuronsPerBlock+MAX_STREAMS, warpSize)*sizeof(short);
}



/* Load firing data for d most recent simulation cycles for all neurons */
//! \todo generalise to deal with longer delays than 32.
__device__
void
loadRecentFiring(int neuronsPerBlock, size_t pitch32, int* gFiring, int* sFiring)
{
	for( int i=0; i < neuronsPerThread(neuronsPerBlock); ++i ){
		if(activeNeuron(i, neuronsPerBlock)){
			sFiring[threadIdx.x + i*THREADS_PER_BLOCK] =
				gFiring[mul24(blockIdx.x, pitch32/4) + threadIdx.x + i*THREADS_PER_BLOCK];
		}
	}
}



/* Store firing data for d most recent simulation cycles */
//! \todo generalise to deal with longer delays than 32.
__device__
void
storeRecentFiring(int neuronsPerBlock, size_t pitch32, int* sFiring, int* gFiring)
{
	for( int i=0; i < neuronsPerThread(neuronsPerBlock); ++i ){
		if(activeNeuron(i, neuronsPerBlock)){
			gFiring[mul24(blockIdx.x, pitch32/4) + threadIdx.x + i*THREADS_PER_BLOCK] =
				sFiring[threadIdx.x + i*THREADS_PER_BLOCK]; 
		}
	}
}



/*! \return Firing status of given neuron at given delay */
__device__
bool
fired(int nn, int delay, int* sFiring)
{
	return sFiring[nn] & (0x1 << (delay-1));
}



/*! update firing buffer at delay 0 */ 
__device__
void
setFiring(int nn, int firing, int maxDelay, int* sFiring)
{
	sFiring[nn] = ( sFiring[nn] << 1 ) | ( firing & 0x1 );
}




//=============================================================================
// Firing delay buffer
//=============================================================================
// The firing delays buffer stores for each pre-synaptic neuron a bit-vector
// (packed into 32b word) specifying at what delays at least one spike reaches
// a post-synaptic array
//=============================================================================

/*! \return size (in bytes) of firing buffer */
__device__ __host__
size_t
firingDelayBufferSize(int neuronsPerBlock, size_t warpSize)
{
	return warpAlign(neuronsPerBlock, warpSize)*sizeof(int);
}



//=============================================================================
// Firing 
//=============================================================================


/*! \todo this might be a bit more straightforward if we just use the same
 * array for internally and externally driven inputs.  */
__device__
void
loadExternalFiring(bool useExternalInput,
		int neuronsPerBlock, 
		uint32_t* gFiring,
		uint32_t* sFiring)
{
	if(threadIdx.x < neuronsPerBlock / 32 + (neuronsPerBlock % 32 ? 1 : 0)) {
		if(useExternalInput) {
			sFiring[threadIdx.x] = gFiring[threadIdx.x];		
		} else {
			sFiring[threadIdx.x] = 0;
		}
	}
	__syncthreads();
}



__device__
void
fire(int time,
	bool externalInput,
	int neuronsPerBlock,
	int maxDelay,
	size_t pitch32,
	float scalingFactor,
	float* gA,  float* gB, float* gC, float* gD,  // input neuron parameters
	// input
	uint32_t* gExtFiring,                            // externally driven firing
	float* sCurrent,                              // input current
	float* gV, float* gU,                         // neuron state
	// buffer
	int* sFiring,
	uint32_t* sExtFiring,
	// output
	int* gFiring)                                // output firing vector
{
	int n = neuronsPerThread(neuronsPerBlock);
	int gIndex = mul24(blockIdx.x, pitch32/4) + threadIdx.x;

	loadExternalFiring(externalInput, neuronsPerBlock, gExtFiring, sExtFiring);
	
	for( int i=0; i<n; ++i ){

		//__syncthreads(); // to ensure global memory accesses are coherent

		if(activeNeuron(i, neuronsPerBlock)) {

			int sIndex = i*THREADS_PER_BLOCK + threadIdx.x;

			float v = gV[gIndex + i*THREADS_PER_BLOCK];
			float u = gU[gIndex + i*THREADS_PER_BLOCK];

			/* Two half-steps are for numerical stability */
			float I = sCurrent[sIndex] * scalingFactor;
			v += 0.5 * (v * (0.04f*v + 5.0f) + 140.0f - u + I);
			v += 0.5 * (v * (0.04f*v + 5.0f) + 140.0f - u + I);
			sCurrent[sIndex] = 0.0f;

			/*! \todo do we save anything by interspersing computation and
			 * memory access? If so, load a and b a bit earlier */
			float a = gA[gIndex + i*THREADS_PER_BLOCK];
			float b = gB[gIndex + i*THREADS_PER_BLOCK];

			u += a * ( b*v - u );

			bool forceFiring = (sExtFiring[sIndex/32] >> (sIndex % 32)) & 0x1;
			char firing = 0;

			//! \todo get threshold from constant memory or from parameter
			if( v >= 30.f || forceFiring) 
			{
				v = gC[gIndex + i*THREADS_PER_BLOCK];
				float d = gD[gIndex + i*THREADS_PER_BLOCK];
				u += d;	
				firing = 1;
			}
			// sync here to ensure coalesced memory access?

			//! \todo use nn instead of sIndex throughout
			setFiring(sIndex, firing, maxDelay, sFiring);
			gV[gIndex+i*THREADS_PER_BLOCK] = v;
			gU[gIndex+i*THREADS_PER_BLOCK] = u;
		}
	}
}





//=============================================================================
// Current buffer
//=============================================================================
// The current buffer stores (in shared memory) the accumulated current for
// each neuron in the block 
//=============================================================================


__device__ __host__
size_t
currentBufferSize(int neuronsPerBlock, size_t warpSize)
{
	return warpAlign(neuronsPerBlock, warpSize)*sizeof(float);
}



/*! Current may be provided by the host. Whether or not this is the case is
 * configured on a per-cluster basis. loadCurrent loads the current buffer with
 * the external input if the current cluster is configured for it.  Otherwise
 * clear. */
__device__
void
loadCurrent(int neuronsPerBlock,
		int neuronsPerThread,
		float* gExtI,
		float* sCurrent)
{
	for( int i=0; i<neuronsPerThread; ++i ){
		int offset = i*THREADS_PER_BLOCK;
		if(activeNeuron(i, neuronsPerBlock)) {
			if(cHasExternalCurrent[blockIdx.x]) {
				sCurrent[threadIdx.x + offset] = gExtI[threadIdx.x + offset];
			} else {
				sCurrent[threadIdx.x + offset] = 0;
			}
		}
	}
}




//=============================================================================
// Weight buffer
//=============================================================================



/*! \return size (in bytes) of weight buffer
 * 
 * The weight buffer stores the weights for up to THREADS_PER_BLOCK
 * post-synaptic neurons. 
 */
__device__ __host__
size_t 
weightBufferSize(size_t warpSize)
{
	return warpAlign(THREADS_PER_BLOCK, warpSize)*sizeof(float);
}



/*! \return size (in bytes) of delays buffer
 *
 * The delays buffer stores the delays for up to THREADS_PER_BLOCK
 * post-synaptic neurons */
__device__ __host__
size_t
delayBufferSize(size_t warpSize)
{
	return warpAlign(THREADS_PER_BLOCK, warpSize);
}



__device__
void
loadWeights(int npt, int npb, size_t pitch32, int pre, float* gWeights, float* sWeights)
{
	for( int i=0; i<npt; ++i ){
		int offset = i*THREADS_PER_BLOCK;
		if(activeNeuron(i, npb)) {
			sWeights[threadIdx.x + offset] = 
				gWeights[  blockIdx.x * npb * pitch32/4   // cluster
				         + pre * pitch32/4                // presynaptic neuron
						 + threadIdx.x + offset];         // postsynaptic neuron
		}
	}
}



__device__
void
loadWeightsChunk(int chunk, int npb, size_t pitch32, int pre,
	float* gWeights, float* sWeights, uchar* sDelays)
{
	int offset = chunk*THREADS_PER_BLOCK;
	if(activeNeuron(chunk, npb)) {
		float w = 
			gWeights[  blockIdx.x * npb * pitch32/4   // cluster
			+ pre * pitch32/4                // presynaptic neuron
			+ threadIdx.x + offset];         // postsynaptic neuron

#ifdef BIT_PACK_DELAYS						
			int bits = *((int*)&w);
			sDelays[threadIdx.x] = bits & 0x1f;
			bits &= ~0x1f;
			sWeights[threadIdx.x] = __int_as_float(bits);
#else
			sWeights[threadIdx.x] = w;
#endif
	}
}



__device__
void
loadDelaysChunk(int chunk, int npb, size_t pitch8, int pre, uchar* gDelays, uchar* sDelays)
{
	int offset = chunk*THREADS_PER_BLOCK;
	if(activeNeuron(chunk, npb)) {
		sDelays[threadIdx.x] = 
			gDelays[   blockIdx.x * npb * pitch8   // cluster
			+ pre * pitch8                // presynaptic neuron
			+ threadIdx.x + offset];      // postsynaptic neuron
	}
}



//=============================================================================
// Integration
//=============================================================================


__device__
short
max(short a, short b) 
{
	return a > b ? a : b;
}


/* The main integrate loop needs to know the indices of the presynaptic neurons
 * which are *likely* to fire this simulation cycle. Iterating through the
 * whole firing array in each thread is too time-consuming. In order to achive
 * both fast presynaptic index generation and lookup we use the following
 * scheme:
 *
 * - we use an index array with one entry for each neuron
 * - array indices are offsets to the next potentially firing neuron
 * - the MSB is reserved to indicate whether the entry is a potentially firing
 *   neuron, rather than just a jump with no associated computation.  
 */
__device__
void
setFiringIdxPar(int neuronsPerBlock, 
		unsigned char maxDelay,
		int streams,
		int* sFiring,
		int* sFiringDelays,
		short* sFiringIdx)
{
	const short VALID_BIT = 0x8000;

	//short* sFiringIdx = sFiringIdxFull[MAX_STREAMS];

	/*! The if condition is not strictly speaking needed, as we check the
	 * presynaptic indices before updating shared memory. The timing is the
	 * same, however, and this way we're less likely to get an overflow of the
	 * indices if we a combination of a large number of threads and a large
	 * chunk size */
	const int base = mul24(threadIdx.x, FIRING_IDX_CHUNK_SIZE);
	if(base < neuronsPerBlock) {

		int next = base + mul24(streams, FIRING_IDX_CHUNK_SIZE);
		for(int presynaptic = base + FIRING_IDX_CHUNK_SIZE - 1;
		        presynaptic > base;
				--presynaptic){
			if(presynaptic < neuronsPerBlock && 
					(sFiring[presynaptic] & (~(~0 << maxDelay)) & sFiringDelays[presynaptic])) { 
				sFiringIdx[presynaptic] = (next - presynaptic) | VALID_BIT;
				next = presynaptic;
			}
		}
		int offset = (next - base);
		if(sFiring[base] & (~(~0 << maxDelay)) & sFiringDelays[base]) { 
			offset |= VALID_BIT;
		}
		sFiringIdx[base] = (short) offset;
	}

	__syncthreads();

	/* Remove redundant entries and create 'chains' number of chains */
	/*! \note this is not quite load-balanced */
	short* sStreamLength = &sFiringIdx[neuronsPerBlock];

	if(threadIdx.x < streams) {

		short current = base;

		/* Set the starting entry */
		while(!(sFiringIdx[current] & VALID_BIT) && current < neuronsPerBlock) {
			current += sFiringIdx[current] & ~VALID_BIT;
		}
		sFiringIdx[-threadIdx.x-1] = current; 

		short next = current + (sFiringIdx[current] & ~VALID_BIT);
		short length = 0;
			
		while(current < neuronsPerBlock) {

			while(!(sFiringIdx[next] & VALID_BIT) && next < neuronsPerBlock) {
				next += sFiringIdx[next] & ~VALID_BIT;
			}

			/* The valid bit is no longer needed, since the table now contains
			 * only valid entries */
			sFiringIdx[current] = next - current;
			++length;
			current = next;
			next += sFiringIdx[current] & ~VALID_BIT;
		}
		sStreamLength[threadIdx.x] = length;
	}

	__syncthreads();
	
	//! \todo perhaps use a tree structure here?
	/* Need the maximum thread length. The thread length is written just beyond
	 * the end of the index array */
	if(threadIdx.x == 0) {
		//! \todo operate directly on shared memory
		short maxLength = 0;
		for(int stream=0; stream<streams; ++stream) {
			maxLength = max(sStreamLength[stream], maxLength);
		}
		sStreamLength[0] = maxLength;
	}
}


__device__
void
integrateDense(int cycle,
	unsigned char maxDelay,         // parameters
	int neuronsPerBlock,
	size_t pitch32, size_t pitch8,
	float* gWeights,                // inputs
#ifndef BIT_PACK_DELAYS
	uchar* gDelays,
#endif
	float* sWeights,                // buffers
	uchar* sDelays,
	int* sFiringDelays,
	short* sFiringIdx,
	int* sFiring,                   // outputs
	float* sCurrent)
{
	int n = neuronsPerThread(neuronsPerBlock);
	setFiringIdxPar(neuronsPerBlock, maxDelay, 1,
			sFiring, sFiringDelays, sFiringIdx);
	__syncthreads();

	for(int presynaptic = sFiringIdx[-1];
			presynaptic < neuronsPerBlock;
			presynaptic += sFiringIdx[presynaptic]){

		for(int postChunk=0; postChunk<n; ++postChunk){

			//! \todo load delay bits here on per (post-)chunk basis
			/* Only if the firing neuron has a spike reaching some post neuron
			 * this cycle should we proceed to loading weights/delays */

			loadWeightsChunk(postChunk, neuronsPerBlock, pitch32,
					presynaptic, gWeights, sWeights, sDelays);
#ifndef BIT_PACK_DELAYS // otherwise load along with weights
			loadDelaysChunk(postChunk, neuronsPerBlock, pitch8,
					presynaptic, gDelays, sDelays);
#endif
			/*! \note if we're careful to leave enough space at the end of
			 * sCurrent, we can remove the conditional for a small speedup */
			unsigned char delay = sDelays[threadIdx.x];
			float weight = sWeights[threadIdx.x];
			if(weight != 0.0f && fired(presynaptic, delay, sFiring)) {
				if(activeNeuron(postChunk, neuronsPerBlock)) {
					int postsynaptic = postChunk * THREADS_PER_BLOCK + threadIdx.x;
					sCurrent[postsynaptic] += weight;
				}
			}
		}
		__syncthreads();
	}
}



__device__
void
integrateSparseSeq(
	unsigned char maxDelay,
	int neuronsPerBlock,
	size_t pitch32,
	int maxColumnIndex,
	//inputs
	int2* gConnectivity,
	int* sFiring,
	int* sFiringDelays,
	// buffers
	short* sFiringIdx,
	// outputs
	float* sCurrent)
{
	setFiringIdxPar(neuronsPerBlock, maxDelay, 1,
			sFiring, sFiringDelays, sFiringIdx);
	__syncthreads();

	//! \todo do pre-chunking for firing delays

	/*! \note This is the maximum number of chunks required for this whole
	 * cluster. It should be possible to reduce this for rows with few
	 * entries. Perhaps better to just save the number of chunks in
	 * constant memory. It would depend on the chunk size, though. */
	int maxChunk = maxColumnIndex / THREADS_PER_BLOCK;

	//! \todo see integrateSparsePar for correct loading
	for(int presynaptic = sFiringIdx[-1];
			presynaptic < neuronsPerBlock;
			presynaptic += sFiringIdx[presynaptic]){

		for(int postChunk=0; postChunk<=maxChunk; ++postChunk) {

			int columnIdx = postChunk * THREADS_PER_BLOCK + threadIdx.x;

			if(columnIdx < cMaxColumnIndex[blockIdx.x]) {

				int2 data = gConnectivity[ presynaptic * pitch32/sizeof(int2) + columnIdx];
				float weight = __int_as_float(data.y);
				short postsynaptic = data.x >> 16;
				uchar delay = data.x & 0xff;
				//! \note some warp divergence here. Can avoid by grouping entries according to delay
				//! \todo ensure that there are no bank conflicts here (must be done on host-side)
				if(weight != 0.0f && fired(presynaptic, delay, sFiring)) { 
					sCurrent[postsynaptic] += weight;	
				}
			} 
		}
	}
}



/* jobSize < THREADS_PER_BLOCK, hence parallel job processing */
__device__
void
integrateSparsePar(int cycle,
	unsigned char maxDelay,
	int neuronsPerBlock,
	size_t pitch32,
	int maxColumnIndex,
	//inputs
	int2* gConnectivity,
	int* sFiring,
	int* sFiringDelays,
	// buffers
	short* sFiringIdx,
	// outputs
	float* sCurrent)
{
	/* A "job" in this context is the computation associated with a single
	 * firing presynaptic neuron with a spike reaching a postsynaptic neuron
	 * this simulation cycle. The size of the job depends on the density of the
	 * connetion matrix. This is deteremined off-line and is set on a
	 * per-cluster basis (rounded to the nearest warp boundary). When possible,
	 * different threads process different jobs. However, if the job size is
	 * larger than number of threads per block, this is not possible, and
	 * instead we need to deal with the jobs in separate chunks (in a different
	 * function). The job size must be at least as large as the warp size in
	 * order to avoid race conditions when updating sCurrent. */

	//! \todo set this in constant memory
	const int jobSize = 64;
	//! \todo get this value back from setFiringIdxPar?
	const int streams = THREADS_PER_BLOCK / jobSize;

	setFiringIdxPar(neuronsPerBlock, maxDelay, streams, 
			sFiring, sFiringDelays, sFiringIdx);

	__syncthreads();

	//! \todo do pre-chunking for firing delays

	int stream = threadIdx.x / jobSize;
	for(int i=0, presynaptic = sFiringIdx[-stream-1];
			i < sFiringIdx[neuronsPerBlock];
			++i, presynaptic += presynaptic > neuronsPerBlock ? 0 : sFiringIdx[presynaptic]) {

		bool write = false;
		short postsynaptic = 0;
		float weight = 0.0f;

		if(presynaptic < neuronsPerBlock) {

			int columnIdx = threadIdx.x % jobSize;
			if(columnIdx < cMaxColumnIndex[blockIdx.x]) {

				//! \todo might be safe to use mul24 here, but depends on npb.
				int2 data = gConnectivity[ presynaptic * pitch32/sizeof(int2) + columnIdx];
				weight = __int_as_float(data.y);
				postsynaptic = data.x >> 16;
				uchar delay = data.x & 0xff;
				if(weight != 0.0f && fired(presynaptic, delay, sFiring)) { 
					write = true;
				}
			} 
		} 
		__syncthreads();

#if 1
		/* Serialised access to sCurrent to avoid race condition. This only
		 * required for devices of compute capability <1.2, which do not have
		 * atomic operations on shared memory */
		/* \todo use atomic operations for devices >1.2 */ 
		for(int s=0; s<streams; ++s) {
			if(s == stream && write) {
				//! \todo ensure that there are no bank conflicts here (must be done on host-side)
				sCurrent[postsynaptic] += weight;	
			}
			__syncthreads();
		}
#endif
	}
}



/*! Combined integrate and fire using sparse connectivity matrix, a single step
* updates the state (u and v) of each neuron and produces spikes to be used in
* the next simulation cycle. 
* 
* The number of neurons per block provided to the kernel is always
* warp-aligned. This means that some threads do useless work, but at no cost.
* Using a warp-aligned neuron number simplifies the control when the number of
* neurons is not an exact multiple of the number of threads per block.
*
 * The parameters (a, b, c, and d) can be set for each individual neuron and
 * should be pre-loaded in global memory before this kernel is invoked.
 * 
 * An external input current can be provided to each neuron in the array gExtI.
 * 
 * The arguments have a fixed lengths of NEURONS_PER_BLOCK, except gWeights
 * which is NEURONS_PER_BLOCK * NEURONS_PER_BLOCK.
 *
 * \todo deal with larger neuron populations.
 */
__global__
void
stepSimulation(int currentCycle,
		int updates,
		int neuronsPerBlock, // not warp aligned
		unsigned char maxDelay,
		float currentScaling,
		bool anySparse,      // at least one cluster uses sparse encoding
		int* gFiring, 
		int* gFiringDelays,
		float* gV, float* gU, 
		float* gExtI,        // externally driven current
		uint32_t* gExtFiring,    // externally driven firing
		float* gConnectivity,
#ifndef BIT_PACK_DELAYS
		unsigned char* gDelays,
#endif
		float* gA, float* gB, float* gC, float* gD,
		size_t pitch32, size_t pitch8)
{
	/* The size of the buffers in shared memory depends on the numbers of
	 * neurons per block, which is only known at run-time. We therefore have to
	 * manually manage this memory 
	 */
	float* sCurrent    = (float*) sMem;
	int* sFiring       = (int*)   &sCurrent      [ currentBufferSize(neuronsPerBlock, warpSize)/sizeof(float)   ];
	int* sFiringDelays =          &sFiring       [ firingBufferSize(neuronsPerBlock, warpSize)/sizeof(int)      ];
	//! \todo the weights and delays are only used in dense encoding
	float* sWeights    = (float*) &sFiringDelays [ firingDelayBufferSize(neuronsPerBlock, warpSize)/sizeof(int) ];
	uchar* sDelays     = (uchar*) &sWeights      [ weightBufferSize(warpSize)/sizeof(float)                     ];
	short* sFiringIdx  = (short*) &sDelays       [ delayBufferSize(warpSize)                                    ];

	//! \todo deal with different number of external inputs and outputs
	//! \todo get the number of external inputs and outputs from somewhere

	loadCurrent(neuronsPerBlock, 
			neuronsPerThread(neuronsPerBlock), gExtI, sCurrent);
	loadRecentFiring(neuronsPerBlock, pitch32, gFiring, sFiring);
	loadRecentFiring(neuronsPerBlock, pitch32, gFiringDelays, sFiringDelays);
	__syncthreads();
	int maxColumnIndex = cMaxColumnIndex[blockIdx.x];

	for( int i=0; i < updates; ++i ){
		if(maxColumnIndex == DENSE_ENCODING) {
			integrateDense(currentCycle, maxDelay,       // parameters
					neuronsPerBlock,
					pitch32, pitch8,
					//! \todo pre-compute part of the address here, rather than in each thread
					gConnectivity,         // inputs
#ifndef BIT_PACK_DELAYS
					gDelays, 
#endif
					sWeights, sDelays,     // buffers
					sFiringDelays, 
					sFiringIdx + MAX_STREAMS,
					sFiring, sCurrent);    // outputs
		} else {
			//! \todo check job size before choosing par or seq
			integrateSparsePar(currentCycle, maxDelay,      // parameters
					neuronsPerBlock,
					pitch32, 
					maxColumnIndex,
					//! \todo might be safe to use mul24 here, but depends on npb.
					((int2*) gConnectivity) + (blockIdx.x * neuronsPerBlock * pitch32/sizeof(int2)), 
					sFiring, 
					sFiringDelays,
					sFiringIdx + MAX_STREAMS,
					sCurrent);             // outputs

		}

		__syncthreads();

		fire(currentCycle,
				i == 0 && gExtFiring != 0,
				neuronsPerBlock, maxDelay, pitch32, 
				currentScaling, 
				gA, gB, gC, gD, 
				gExtFiring, sCurrent, 
				gV, gU,
				// buffer
				sFiring, 
				(uint32_t*) sFiringIdx, // re-purpose while not in use
				// output
				gFiring);
		__syncthreads();
	}

	storeRecentFiring(neuronsPerBlock, pitch32, sFiring, gFiring);
}




/* The shared memory size depends on the number of neurons per block, which is
 * only known at run time. We therefore need to handle this memory manually.
 * However, to avoid warp divergence and bank conflicts it's desirable that the
 * arrays in shared memory is aligned to warp boundaries
 */
__host__
size_t
sharedMemorySize(int neurons, int maxDelay, int warpSize, bool sparseEncoding)
{
	return currentBufferSize(neurons, warpSize)
	     + firingBufferSize(neurons, warpSize)
		 + firingDelayBufferSize(neurons, warpSize)
		 + weightBufferSize(warpSize)
		 + delayBufferSize(warpSize)
		 + firingIdxBufferSize(neurons, warpSize);
}




/*! Wrapper for the __global__ call that performs a single simulation step */
__host__
KernelError
step(cudaDeviceProp* deviceProperties,
		int currentCycle,
		int updates, 
		DeviceMemory* gMem, 
		float currentScaling,
		const float* extI, 
		const uint32_t* extFiring,
		int* firing, 
		float* v,
		int probe)
{
	if(gMem->clusterCount() > MAX_THREAD_BLOCKS) {
		return KERNEL_MAX_CLUSTERS_EXCEEDED;	
	}
	//! \todo double buffer the external stimulus and transfer this asynchrounously

	/* Transfer external stimulus */
	if(extI != NULL)
		cudaMemcpy(gMem->extI, extI, gMem->n*sizeof(float), cudaMemcpyHostToDevice);

	//! \todo assert that if extFiring is NULL, no cluster is configured to
	//read this information. Ditto for external current.
	if(extFiring != NULL)
		cudaMemcpy(gMem->extFiring, 
				extFiring,
				gMem->pitch1,
				cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_PER_BLOCK, 1);
	dim3 dimGrid(gMem->clusterCount(), 1);

	size_t sharedMemSize = sharedMemorySize(gMem->n,
			gMem->maxDelay(),
			deviceProperties->warpSize,
			gMem->sparseEncoding());

	if(currentCycle == 0) {
		fprintf(stderr, "Shared mem: %dB\n", sharedMemSize);
	}

	if(sharedMemSize > deviceProperties->sharedMemPerBlock) {
		fprintf(stderr, "Insufficient shared memory");
		return KERNEL_INSUFFICIENT_SHARED_MEMORY;
	}

	stepSimulation<<<dimGrid, dimBlock, sharedMemSize>>>(
			currentCycle,
			updates, 
			gMem->n,
			gMem->maxDelay(),
			currentScaling,
			gMem->sparseEncoding(),
			gMem->firing, gMem->firingDelays,
			gMem->v, gMem->u, 
			gMem->extI, 
			extFiring ? gMem->extFiring : 0, 
			gMem->weights,
#ifndef BIT_PACK_DELAYS
			gMem->delays,
#endif
			gMem->a, gMem->b, gMem->c, gMem->d,
			gMem->pitch32, gMem->pitch8);

	cudaError_t status = cudaGetLastError();
	if(status != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(status));
		fprintf(stderr, "Shared memory size: %d\n", sharedMemSize);
		return KERNEL_CUDA_ERROR;
	}

	if(firing != NULL) {
		int* gFiring = gMem->firingAddress(probe);
		if(gFiring != NULL) {
			cudaMemcpy(firing, gFiring, gMem->n*sizeof(int), cudaMemcpyDeviceToHost);
		} else {
			fprintf(stderr, "Invalid probe specified\n");
			return KERNEL_INVALID_PROBE;
		}
	}

	if(v != NULL) {
		float* gV = gMem->vAddress(probe);
		if(gV != NULL) {
			cudaMemcpy(v, gV, gMem->n*sizeof(float), cudaMemcpyDeviceToHost);
		} else {
			fprintf(stderr, "Invalid probe specified\n");
			return KERNEL_INVALID_PROBE;
		}
	}

	return KERNEL_OK;
}




__host__
KernelError
configureClusters(const char* externalCurrent,
	const char* externalFiring,
	const int* maxColumnIndex)
{
	cudaError_t status;
	if((status = cudaMemcpyToSymbol(cHasExternalCurrent,
			externalCurrent,
			MAX_THREAD_BLOCKS, 
			0,
			cudaMemcpyHostToDevice)) != cudaSuccess){
		fprintf(stderr, "%s\n", cudaGetErrorString(status));
		return KERNEL_CONSTANT_MEMORY_ERROR;
	}

	if((status = cudaMemcpyToSymbol(cHasExternalFiring,
			externalFiring,
			MAX_THREAD_BLOCKS, 
			0,
			cudaMemcpyHostToDevice)) != cudaSuccess){
		fprintf(stderr, "%s\n", cudaGetErrorString(status));
		return KERNEL_CONSTANT_MEMORY_ERROR;
	}

	if((status = cudaMemcpyToSymbol(cMaxColumnIndex,
			maxColumnIndex,
			MAX_THREAD_BLOCKS*sizeof(int), 
			0,
			cudaMemcpyHostToDevice)) != cudaSuccess){
		fprintf(stderr, "%s\n", cudaGetErrorString(status));
		return KERNEL_CONSTANT_MEMORY_ERROR;
	}

	return KERNEL_OK;
}
