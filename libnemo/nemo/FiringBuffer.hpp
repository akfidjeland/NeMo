#ifndef NEMO_FIRING_BUFFER_HPP
#define NEMO_FIRING_BUFFER_HPP

#include <deque>
#include <vector>
#include <nemo/types.h>

namespace nemo {


/* One cycle's worth of fired neurons. Note that we store a reference to the
 * vector of neurons, so the user needs to be aware of the lifetime of this and
 * copy the contents if appropriate. */
struct FiredList
{
	cycle_t cycle;
	std::vector<unsigned>& neurons;

	FiredList(cycle_t cycle, std::vector<unsigned>& neurons) :
		cycle(cycle), neurons(neurons) { }
};


/* A firing buffer containing a FIFO of firing data with one entry for each
 * cycle. */
class FiringBuffer
{
	public :

		FiringBuffer();

		/*! Add a new cycle at the end of the FIFO */
		std::vector<unsigned>& enqueue();

		/*! Discard the current oldest cycle's data and return reference to the
		 * new oldest cycle's data. The data referenced in the returned list of
		 * firings is valid until the next call to \a read or \a dequeue. */
		FiredList dequeue();

	private :

		std::deque< std::vector<unsigned> > m_fired;

		cycle_t m_oldestCycle;
};

}

#endif
