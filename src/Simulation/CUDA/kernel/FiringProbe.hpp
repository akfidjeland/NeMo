#ifndef FIRING_PROBE_HPP
#define FIRING_PROBE_HPP

//! \file FiringProbe.hpp

/*! \brief Data and functions for reading firing data from device to host
 *
 * \author Andreas Fidjeland
 */


#include <cuda_runtime.h>
#include "firingProbe.cu_h"
#include <vector>



/* Device and host data for dealing with firing output */
struct FiringProbe
{
	public :

		/*! Set up data on both host and device for probing firing
		 * \param partitionSize
		 *		This is the maximum number of neurons in a partition.
		 * \param maxReadPeriod
		 *		The maximum period between reads issued from the host. If this
		 *		period is exceeded, a buffer overflow is possible.  Whether or
		 *		not this will happen depends on the firing rate in the network.
		 *		The frequency is determined in terms of steps of whatever
		 *		temporal resolution the simulation is running at.
		 */
		FiringProbe(size_t partitionCount,
				size_t partitionSize,
				uint maxReadPeriod);

		~FiringProbe();

		/*! Read all firing data buffered on the device since the previous call
		 * to this function (or the start of simulation if this is the first
		 * call). The return vectors are valid until the next call to this
		 * function. */
		//! \todo require min and max expected cycle number to detect possible overflow
		void readFiring(
				uint** cycles,
				uint** partitionIdx,
				uint** neuronIdx,
				size_t* len);

		ushort2* deviceBuffer() const;

		/* As the buffer fills up, each partition needs to keep track of the
		 * next free buffer entry. Since the buffer is used across kernel
		 * invocation, we need to keep track of this in global memory. The data
		 * is the offset (in words) to the beginning of the next free row */
		uint* deviceNextFree() const;

		/*! \return size of host buffer (required for testing only) */
		size_t hostBufferSize() const;

		uint maxReadPeriod() const;

        /*! The user is in control of when to read the buffer. This opens up
         * the possibility of overflowing the buffers through negligence on the
         * part of the caller. To avoid this, keep an internal counter counting
         * down to the time we can no longer guarantee no overflow. If the user
         * has not emptied the buffer by then, clear the buffers, discarding
         * the firing data. */
        void checkOverflow();

	private :

		/* On the device side, the firing is stored as interleaved
		 * per-partition streams. Each stream contains a number of chunks, the
		 * size of which is hard-coded in the macro FMEM_CHUNK_SIZE user. It
		 * should be at least the size of a warp to avoid bank conflicts, and
		 * can be sized up to the number of threads. A single partition has
		 * exclusive access to every nth such chunk, where n is the number of
		 * partitions. With this scheme the amount of data that needs to be
		 * read back from the host is limited by the partition with the highest
		 * firing rate. A denser scheme would require interaction between the
		 * partitions.
		 * 
		 * For each firing we need to store the time and the neuron index
		 * within a partition. The partition index is implicit in the data
		 * structure. We use 16-bit values for these two fields, which allows
		 * partitions of up to 32k neurons and up buffering of up to 32k cycles
		 * worth of firing.  */
		ushort2* m_deviceBuffer;

		/* Pitch (in words) between each partition-specific entry */
		size_t m_ppitch;

		/*! \todo may want to combine all such per-partition variables and
		 * allow them to accessed in a single coalesced read operation */
		uint* m_deviceNextFree;

		void resetNextFree();

		/* Host side buffer for staging output. This is fixed-size and pinned
		 * for better bandwidth. */
		ushort2* m_hostBuffer;
		size_t m_hostBufferSize; // in words

		std::vector<uint> setStreamLength();

		uint fillOutputBuffer(size_t maxFirings,
				const std::vector<uint>& chunksPerStream);

		/* Host side buffers for ordered output. The data when read back is
		 * unordered. The ordered data is written to host_ordered. This data is
		 * resized when required, and pointers to this data are valid untill
		 * the next read operation */
		std::vector<uint> m_hostCycles;
		std::vector<uint> m_hostPartitionIdx;
		std::vector<uint> m_hostNeuronIdx;

		void resizeOutputBuffer(size_t maxFirings);

		size_t m_partitionCount;

		/* After a certain number of cycles, a buffer overflow may occur */
		uint m_maxReadPeriod;

        uint m_nextOverflow;
};


#endif
