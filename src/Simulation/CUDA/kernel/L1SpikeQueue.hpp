#ifndef L1_SPIKE_QUEUE_HPP
#define L1_SPIKE_QUEUE_HPP

#include <cuda_runtime.h>
#include <cstddef>

struct L1SpikeQueue
{
	public :

		/*! 
		 * \param partitionCount
		 * 		Number of distinct partition in the network
		 * \param entrySize
		 * 		Maximum number of synapses connecting any pair of partitions
		 */
		L1SpikeQueue(size_t partitionCount, size_t entrySize, size_t l1pitch);

		~L1SpikeQueue();

		/*! Each entry in the queue contains synaptic data packed in 64-bit words
		 * containing postsynaptic index (2 bytes), padding (2 * bytes), and weight
		 * (4 bytes).  */
		//! \todo make this unsigned
		uint2* data() const;

		/* width in words of each delay entry, including padding */
		size_t pitch() const;

		/*! In order to keep a dense queue while avoiding writing twice to the
		 * same location.  we maintain a list of queue head indices, specifying
		 * which word in the entry is the next free one. Since the source does
		 * random access writes, but the target needs to clear a number of
		 * entries when emptying the queue, the queue head data is organised
		 * such that all the entries for a single target can be cleared in a
		 * single sweep. Thus the indexing is: head[target][delay][source], and
		 * for a given current entry the target can clear head[target][d] */
		//! \todo could probably do with short here
		unsigned int* heads() const;

		/*! Pitch (in words) between each partitions head entries */
		size_t headPitch() const;

	private :

		uint2* m_data;

		size_t m_pitch;

		unsigned int* m_heads;

		size_t m_headPitch;
};

#endif
