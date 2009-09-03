#ifndef FIRING_OUTPUT_HPP
#define FIRING_OUTPUT_HPP

#include <vector>

//! \file FiringOutput.hpp

/*! \brief Data and functions for reading firing data from device to host
 *
 * \author Andreas Fidjeland
 */

class FiringOutput {

	public:

		/*! Set up data on both host and device for probing firing
		 *
		 * \param partitionSize
		 *		This is the maximum number of neurons in a partition.
		 * \param maxReadPeriod
		 *		The maximum period between reads issued from the host. If this
		 *		period is exceeded, a buffer overflow will occur. Whether or
		 *		not this will happen depends on the firing rate in the network.
		 *		The frequency is determined in terms of steps of whatever
		 *		temporal resolution the simulation is running at.
		 */
		FiringOutput(size_t partitionCount, size_t partitionSize, uint maxReadPeriod);

		~FiringOutput();

		/*! Read all firing data buffered on the device since the previous call
		 * to this function (or the start of simulation if this is the first
		 * call). The return vectors are valid until the next call to this
		 * function. */
		//! \todo require min and max expected cycle number to detect possible overflow
		void readFiring(
				uint** cycles,
				uint** partitionIdx,
				uint** neuronIdx,
				uint* len,
				uint* totalCycles);

		/*! Flush the buffer. Any data on the device is left there as garbage,
		 * so there's no significant cost to doing this. */
		void flushBuffer();

		uint32_t* deviceData() const { return md_buffer; }

		/*! The user is in control of when to read the buffer. This opens up
		 * the possibility of overflowing the buffers through negligence on the
		 * part of the caller. To avoid this, keep an internal counter counting
		 * down to the time we can no longer guarantee no overflow. If the user
		 * has not emptied the buffer by then, clear the buffers, discarding
		 * the firing data.
		 *
		 * We also keep track of how many cycles worth of firing data we have
		 * currently buffered. This is not apparent in the sparse firing data
		 * we return.
		 *
		 * \a step() should be called every simulation cycle to update these
		 * counters.
		 *
		 * \return next (unfilled) cycle's firing buffer
		 */
		uint32_t* step();

		/*! \return bytes of allocated device memory */
		size_t d_allocated() const { return md_allocated; }

		size_t wordPitch() const { return m_pitch; }

	private:

		/* Dense firing buffers on device and host */
		uint32_t* md_buffer;
		uint32_t* mh_buffer; // pinned, same size as device buffer
		size_t m_pitch; // in words

		size_t m_partitionCount;

		/* While the firing is stored in a dense format on the device, the
		 * external interface uses sparse firing. Pointers into the sparse
		 * storage is valid from one call to \a readFiring to the next */
		std::vector<uint> m_cycles;
		std::vector<uint> m_partitionIdx;
		std::vector<uint> m_neuronIdx;

		/*! \see step() */
		uint m_bufferedCycles;
		uint m_maxReadPeriod;

		size_t md_allocated;

		void populateSparse(
				uint cycles,
				const uint32_t* hostBuffer,
				std::vector<uint>& cycles,
				std::vector<uint>& partitionIdx,
				std::vector<uint>& neuronIdx);
};

#endif
