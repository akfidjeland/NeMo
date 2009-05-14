#include <list>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cutil.h>
#include <assert.h>
//! \todo remove this again
#include <stdio.h>

#include "FiringProbe.hpp"
#include "util.h"
#include "log.hpp"



/*! Allocate the host-side buffer for staging data between host and output
 * buffers. This intermediate stage is needed since on the device buffer the
 * data from different partitions are interleaved.
 *
 * The host buffer uses pinned memory since this is the main run-time buffer
 * receiving data from the host and high memory bandwidth is therefore highly
 * desirable. 
 *
 * Since the buffer uses pinned memory -- which is scarce -- the buffer should
 * ideally be small. On the other hand issuing a memory read from the device is
 * costly. We therefore aim to make this buffer large enough that in the
 * average case a single read operation is sufficient. This can be set based on
 * 1) a maximum read rate and 2) an maximum firing rate we want to process with
 * a single read. 
 *
 * \param partitionPitch
 * 		number of words between two consecutive chunks in a single partition's
 * 		stream 
 * \param maxReadPeriod
 * 		maximum number of cycles between each device read
 * \param commonFiringPeriod
 * 		maximum average number of cycles between each neuron firing for which
 * 		reading back the device memory can be done in a single operation. 
 * \param len
 * 		(output) length (in words) of the allocated array
 *
 * \return
 * 		pointer to allocated data
 */
ushort2*
allocHostBuffer(
		size_t partitionCount,
		size_t partitionSize,
		size_t partitionPitch,
		uint maxReadPeriod,
		uint commonFiringPeriod,
		size_t* len)
{
	const size_t avgFiringsPerRead = 
		partitionCount * partitionSize * maxReadPeriod / commonFiringPeriod;

	/* We set the buffer size to be large enough to guranteed be large enough
	 * for one cycle's worth of firing. The only reason for doing this is to
	 * simplify unit tests. */
	const size_t maxFiringsPerCycle = partitionCount * partitionSize;

	const size_t bufferSize = std::max(avgFiringsPerRead, maxFiringsPerCycle); 

	/* The buffer size should be alligned to partition pitch boundary */
	*len = ALIGN(bufferSize, partitionPitch);

	ushort2* arr;
	CUDA_SAFE_CALL(cudaMallocHost((void**) &arr, *len * sizeof(ushort2)));
	return arr;
}



FiringProbe::FiringProbe(
		size_t partitionCount,
		size_t partitionSizeUnaligned,
		uint maxReadPeriod) :
	m_hostBufferSize(0),
	m_partitionCount(partitionCount),
	m_maxReadPeriod(maxReadPeriod),
    m_nextOverflow(maxReadPeriod)
{
	size_t width = FMEM_CHUNK_SIZE * sizeof(ushort2);
    size_t partitionSize = (size_t) ceilPowerOfTwo((uint32_t) partitionSizeUnaligned);

	/* In the worst case every neuron in a partition fires every cycle */
	size_t maxPartitionFirings = partitionSize * sizeof(ushort2);
	size_t height = maxReadPeriod * m_partitionCount * maxPartitionFirings / width;

	size_t bpitch;
	//fprintf(stderr, "Firing device memory: %u x %u\n", height, width);
	CUDA_SAFE_CALL(cudaMallocPitch((void**)(&m_deviceBuffer), &bpitch, width, height));

	/* kernel relies on hard-coded pitch... */
	assert(bpitch ==  FMEM_CHUNK_SIZE*sizeof(ushort2));
	m_ppitch = m_partitionCount * FMEM_CHUNK_SIZE;

	/* Device data containing next free entry for each partition */
	CUDA_SAFE_CALL(cudaMalloc((void**) &m_deviceNextFree, m_partitionCount*sizeof(uint)));
	resetNextFree();

	m_hostBuffer = allocHostBuffer(
			m_partitionCount,
			partitionSize,
			m_ppitch,
			maxReadPeriod,
			30, // approx 30Hz at 1ms precision
			&m_hostBufferSize);
}



FiringProbe::~FiringProbe()
{
	cudaFreeHost(m_hostBuffer);
	cudaFree(m_deviceNextFree);
	cudaFree(m_deviceBuffer);
}



ushort cycle(ushort2* v) { return v->x; }
ushort idx(ushort2* v) { return v->y; }



struct Stream 
{
	Stream(int sidx, size_t chunks) :
		stream(sidx), data(NULL), remainingChunks(chunks) {}

	int stream;

	/*! pointer to next chunk of data to process */
	ushort2* data;

	/*! number of chunks remaining (either overall or within a load) */
	size_t remainingChunks;

	int cycle() { return data->x; }
};



/* Insert entry ordered from lowest to highest cycle. 
 *
 * pre: streams is already sorted
 */
void
insertSorted(std::list<Stream>& streams, Stream candidate)
{
	for(std::list<Stream>::iterator s=streams.begin();
			s != streams.end(); ++s) {
		if(candidate.cycle() < s->cycle()) {
			streams.insert(s, candidate);
			return;
		}
	}
	streams.insert(streams.end(), candidate);
}




/*! Create a sorted (by cycle) list of streams for the current load 
 *
 * \param fmem
 * 		Firing memory data on both host and device
 * \param streamLengths
 * 		Length (in words) of each stream from first to last entry. Since the
 * 		streams are interleaved only a part of the data between the first and
 * 		last entry belongs to any one stream.
 */
std::list<Stream>
initStreams(const std::vector<uint>& chunksPerStream)
{
	std::list<Stream> streams;

	for(std::vector<uint>::const_iterator i=chunksPerStream.begin();
			i != chunksPerStream.end(); ++i) {
		streams.push_back(Stream(i - chunksPerStream.begin(), *i));
	}
	return streams;
}


/* Initialse the streams for the current load. Modify the streams data for all
 * loads to reflect the processing in the current load. */
std::list<Stream>
loadStreams(
		size_t chunksPerStreamAndLoad,
		ushort2* h_buffer,
		size_t chunkSize,
		std::list<Stream>* streams)
{
	std::list<Stream> ret;
	for(std::list<Stream>::iterator i = streams->begin(); 
			i != streams->end(); ++i) {
		Stream candidate = *i;
		candidate.data = h_buffer + candidate.stream * chunkSize;
		candidate.remainingChunks =
			std::min(chunksPerStreamAndLoad, i->remainingChunks);
		if(candidate.remainingChunks > 0) {
			insertSorted(ret, candidate);
			i->remainingChunks -= candidate.remainingChunks;
		}
	}
	return ret;
}


uint 
processStreams(
		size_t ppitch,
		size_t cpitch,
		std::list<Stream>& streams,
		uint** cp,
		uint** pidxp,
		uint** nidxp)
{
	uint firings = 0;
	while(!streams.empty()) {
		Stream next = streams.front();
		streams.pop_front();
		for(ushort2* i=next.data; i<next.data + cpitch; ++i) {
			if(idx(i) == INVALID_NEURON)
				break;
			*(*cp)++    = cycle(i);
			*(*nidxp)++ = idx(i);
			*(*pidxp)++ = next.stream;
			++firings;
		}

		next.data += ppitch;
		next.remainingChunks -= 1;

		if(next.remainingChunks > 0) {
			insertSorted(streams, next);
		}
	}
	return firings;
}



uint
FiringProbe::fillOutputBuffer(
		size_t maxFirings,
		const std::vector<uint>& chunksPerStream)
{
	// output pointers
	uint* cp = &m_hostCycles[0];
	uint* pidxp = &m_hostPartitionIdx[0];
	uint* nidxp = &m_hostNeuronIdx[0];

	assert(m_hostBufferSize >= FMEM_CHUNK_SIZE);
	size_t bufferChunks = m_hostBufferSize / FMEM_CHUNK_SIZE;
	assert(bufferChunks % m_partitionCount == 0);
	size_t chunksPerStreamAndLoad = bufferChunks / m_partitionCount;
	size_t bufferLoads = DIV_CEIL(maxFirings, m_hostBufferSize);

	std::list<Stream> streams =
		initStreams(chunksPerStream);

	uint totalFirings = 0;

	for(size_t load=0; load < bufferLoads; ++load) {

		/* read enough data to fill the host buffer */
		CUDA_SAFE_CALL(cudaMemcpy(m_hostBuffer,
					m_deviceBuffer + load * m_hostBufferSize,
					m_hostBufferSize * sizeof(ushort2),
					cudaMemcpyDeviceToHost));

		std::list<Stream> lstreams = 
			loadStreams(chunksPerStreamAndLoad, m_hostBuffer, FMEM_CHUNK_SIZE, &streams);
		totalFirings += processStreams(m_ppitch, FMEM_CHUNK_SIZE, lstreams, &cp, &pidxp, &nidxp);
	} 

	return totalFirings;
}



/*! (Load and) determine the length of each stream
 *
 * The firing buffer contains interleaved streams. A per-stream word contains
 * the word offset to the next free entry. From the beginning and end addresses
 * of each stream we can compute the number of chunks of data in each stream.
 *
 * \return 
 * 		vector containing number of chunks per stream
 *
 * \todo put the next entries at the beginning of each stream. Then in the
 * common case we can get away with a single read. Perform this single read
 * before calling setStreamLength.
 */
std::vector<uint>
FiringProbe::setStreamLength()
{
	std::vector<uint> ret(m_partitionCount);
	CUDA_SAFE_CALL(cudaMemcpy(&ret[0], m_deviceNextFree, 
				m_partitionCount*sizeof(uint), cudaMemcpyDeviceToHost));
	// ret now contains per-stream max offsets	
	
	for(std::vector<uint>::iterator i=ret.begin(); i!=ret.end(); ++i) {
		size_t maxOffset = *i;
		*i = maxOffset / FMEM_CHUNK_SIZE / m_partitionCount;
	}

	// ret now contains the number of chunks per stream	
	return ret;
}



void
FiringProbe::resetNextFree()
{
	/* Each partition starts at some offset from the base pointer */
	std::vector<uint> startingOffsets(m_partitionCount);
	for(size_t p=0; p<m_partitionCount; ++p) {
		startingOffsets[p] = p * FMEM_CHUNK_SIZE;
	}

	CUDA_SAFE_CALL(cudaMemcpy(m_deviceNextFree,
				&startingOffsets[0],
				m_partitionCount*sizeof(uint),
				cudaMemcpyHostToDevice));

    /* We only reset the headers here, rather than the firing buffers
     * themselves. Some garbage is thus left, but this will not be read later
     */
    m_nextOverflow = m_maxReadPeriod;
}



/*! Re-allocate output buffer such that it can at least contain the current
 * contents of the device buffer */
void
FiringProbe::resizeOutputBuffer(size_t maxFirings)
{
	/* We won't know exactly how many firings there were until we have merged
	 * the data from the different partitions, but this should be sufficient to
	 * deal with every partition's stream being completely filled with firing
	 * data */
	m_hostCycles.resize(maxFirings);
	m_hostPartitionIdx.resize(maxFirings);
	m_hostNeuronIdx.resize(maxFirings);
}



void
FiringProbe::readFiring(uint** cycles, uint** pidx, uint** nidx, size_t* len)
{
	std::vector<uint> chunksPerStream = setStreamLength();
	resetNextFree();
	size_t totalChunks = 
			std::accumulate(chunksPerStream.begin(), chunksPerStream.end(), 0);
	size_t maxFirings = totalChunks * FMEM_CHUNK_SIZE;
	resizeOutputBuffer(maxFirings);
	/* Load data, in several chunks if required */
	*len = fillOutputBuffer(maxFirings, chunksPerStream);
	*cycles = &m_hostCycles[0];
	*nidx = &m_hostNeuronIdx[0];
	*pidx = &m_hostPartitionIdx[0];
}



ushort2*
FiringProbe::deviceBuffer() const
{
	return m_deviceBuffer;
}



uint*
FiringProbe::deviceNextFree() const
{
	return m_deviceNextFree;
}



size_t 
FiringProbe::hostBufferSize() const
{
	return m_hostBufferSize;
}


uint
FiringProbe::maxReadPeriod() const
{
	return m_maxReadPeriod;
}



void
FiringProbe::checkOverflow()
{
    if(m_nextOverflow == 0) {
        WARNING("Emptied device firing buffers to avoid buffer overflow. Firing data lost. Read data from device more frequently to avoid");
        resetNextFree(); // also reset m_nextOverflow;
    }
    m_nextOverflow -= 1;
}
