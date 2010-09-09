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
#include <set>

#include <boost/tuple/tuple.hpp>
#include <boost/shared_array.hpp>
#include <boost/optional.hpp>

#include <nemo/config.h>
#include "types.hpp"
#include "Mapper.hpp"
#include "STDP.hpp"

#define ASSUMED_CACHE_LINE_SIZE 64

namespace nemo {


/* The AxonTerminal in types.hpp includes 'plastic' specification. It's not
 * needed here. */
template<typename W>
struct FAxonTerminal
{
	FAxonTerminal(W w, nidx_t t) : weight(w), target(t) {}

	W weight;
	nidx_t target; 
};



/* A row contains a number of synapses with a fixed source and delay. A
 * fixed-point format is used internally. The caller needs to specify the
 * format.  */
struct Row
{
	Row() : len(0) {}

	/* \post synapse order is the same as in input vector */
	Row(const std::vector<AxonTerminal>&, unsigned fbits);

	size_t len;
	boost::shared_array< FAxonTerminal<fix_t> > data;
};



class NetworkImpl;
class ConfigurationImpl;
struct AxonTerminalAux;


/* Generic connectivity matrix
 *
 * Data in this class is organised for optimal cache performance. A
 * user-defined fixed-point format is used.
 */
class NEMO_BASE_DLL_PUBLIC ConnectivityMatrix
{
	public:

		typedef Mapper<nidx_t, nidx_t> mapper_t;

		 ConnectivityMatrix(const ConfigurationImpl& conf, const mapper_t&);

		/*! Populate runtime CM from existing network.
		 *
		 * The mapper can translate neuron indices (both source and target)
		 * from one index space to another. All later accesses to the CM data
		 * are assumed to be in terms of the translated indices.
		 *
		 * 'finalize' must be called prior to use. This slightly clumsy
		 * interface is there so that we can ensure that the mapper will have a
		 * complete list of valid neuron indices by the time of finalization,
		 * so that we can report invalid synapse terminals.
		 */
		ConnectivityMatrix(
				const NetworkImpl& net,
				const ConfigurationImpl& conf,
				const mapper_t&);

		/*! Add a number of synapses with the same source and delay. Return
		 * reference to the newly inserted row.
		 *
		 * The mapper is used to map the target neuron indices (source indices
		 * are unaffected) from one index space to another.
		 */
		//! \todo remove mapper argument. Mapper is already a member so no need to pass it in.
		Row& setRow(nidx_t source, delay_t,
				const std::vector<AxonTerminal>&,
				const mapper_t&);

		/*! \return all synapses for a given source and delay */
		const Row& getRow(nidx_t source, delay_t) const;

		/*! \copydoc nemo::Simulation::getTargets */
		const std::vector<unsigned>& getTargets(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getDelays */
		const std::vector<unsigned>& getDelays(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getWeights */
		const std::vector<float>& getWeights(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getPlastic */
		const std::vector<unsigned char>& getPlastic(const std::vector<synapse_id>&);

		void finalize(const mapper_t&);

		typedef std::set<delay_t>::const_iterator delay_iterator;

		delay_iterator delay_begin(nidx_t source) const;
		delay_iterator delay_end(nidx_t source) const;

		unsigned fractionalBits() const { return m_fractionalBits; }

		delay_t maxDelay() const { return m_maxDelay; }

		void accumulateStdp(const std::vector<uint64_t>& recentFiring);

		void applyStdp(float reward);

	private:

		const mapper_t& m_mapper;

		unsigned m_fractionalBits;

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
		void finalizeForward(const mapper_t&);

		/* For the reverse matrix we don't need to group by delay */
		//! \todo move into std::vector when finalizing
		typedef std::vector<RSynapse> Incoming;
		std::map<nidx_t, Incoming> m_racc;
		boost::optional<StdpProcess> m_stdp;

		std::map<nidx_t, std::set<delay_t> > m_delays;

		//! \todo could add a fast lookup here as well

		delay_t m_maxDelay;

		/*! \return linear index into CM, based on 2D index (neuron,delay) */
		size_t addressOf(nidx_t, delay_t) const;

		void verifySynapseTerminals(fidx idx, const Row& row, const mapper_t&) const;

		/*! \return address of the synapse weight in the forward matrix, given
		 * a synapse in the reverse matrix */
		fix_t* weight(const RSynapse&) const;

		/* Internal buffers for synapse queries */
		std::vector<unsigned> m_queriedTargets;
		std::vector<unsigned> m_queriedDelays;
		std::vector<float> m_queriedWeights;
		std::vector<unsigned char> m_queriedPlastic;

		/* Additional synapse data which is only needed for runtime queries.
		 * This is kept separate from m_cm so that we can make m_cm fast and
		 * compact. The query information is not crucial for performance.  */
		typedef std::vector<AxonTerminalAux> aux_row;
		typedef std::map<nidx_t, aux_row> aux_map;
		aux_map m_cmAux;

		/* Look up auxillary synapse data and report invalid lookups */
		const AxonTerminalAux& axonTerminalAux(nidx_t neuron, id32_t synapse) const;
		const AxonTerminalAux& axonTerminalAux(const synapse_id&) const;
};



inline
size_t
ConnectivityMatrix::addressOf(nidx_t source, delay_t delay) const
{
	return source * m_maxDelay + delay - 1;
}



/* The parts of the synapse data is only needed if querying synapses at
 * run-time. This data is stored separately */
struct AxonTerminalAux
{
	/* We need to store the synapse address /within/ a row. The row number
	 * itself can be computed on-the-fly based on the delay. */
	sidx_t idx;

	unsigned delay;
	bool plastic;

	AxonTerminalAux(sidx_t idx, unsigned delay, bool plastic) :
		idx(idx), delay(delay), plastic(plastic) { }

	AxonTerminalAux() :
		idx(~0), delay(~0), plastic(false) { }
};



} // end namespace nemo

#endif
