#ifndef NEMO_CONNECTIVITY_MATRIX_HPP
#define NEMO_CONNECTIVITY_MATRIX_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <map>

#include <boost/tuple/tuple.hpp>
#include <boost/shared_array.hpp>

#include "types.hpp"

#define ASSUMED_CACHE_LINE_SIZE 64

namespace nemo {


/* The AxonTerminal in types.hpp includes 'plastic' specification. It's not
 * needed here. */
struct FAxonTerminal
{
	FAxonTerminal(weight_t w, nidx_t t) : weight(w), target(t) {}

	weight_t weight; 
	nidx_t target; 
};



/* A row contains a number of synapses with a fixed source and delay */
struct Row
{
	Row() : len(0) {}

	Row(const std::vector<AxonTerminal<nidx_t, weight_t> >& );

	size_t len;
	boost::shared_array<FAxonTerminal> data;
};


/* Generic connectivity matrix
 *
 * Data in this class is organised for optimal cache performance
 */
class ConnectivityMatrix
{
	public:

		ConnectivityMatrix();

		/*! Add a number of synapses with the same source and delay. Return
		 * reference to the newly inserted row. */
		Row& setRow(nidx_t, delay_t,
				const std::vector<AxonTerminal<nidx_t, weight_t> >&);

		/*! \return all synapses for a given source and delay */
		const Row& getRow(nidx_t source, delay_t) const;

		void finalize() { finalizeForward(); }

	private:

		/* During network construction we accumulate data in a map. This way we
		 * don't need to know the number of neurons or the number of delays in
		 * advance */
		typedef boost::tuple<nidx_t, delay_t> fidx;
		std::map<fidx, Row> m_acc;

		/* At run-time, however, we want the fastest possible lookup of the
		 * rows. We therefore use a vector with linear addressing. This just
		 * points to the data in the accumulator. This is constructed in \a
		 * finalize which must be called prior to getRow being called */
		//! \todo use two different classes for this in order to ensure ordering
		std::vector<Row> m_cm;
		void finalizeForward();

		delay_t m_maxDelay;
		nidx_t m_maxSourceIdx;

		/*! \return linear index into CM, based on 2D index (neuron,delay) */
		size_t addressOf(nidx_t, delay_t) const;
};



inline
size_t
ConnectivityMatrix::addressOf(nidx_t source, delay_t delay) const
{
	return source * m_maxDelay + delay - 1;
}

} // end namespace nemo

#endif
