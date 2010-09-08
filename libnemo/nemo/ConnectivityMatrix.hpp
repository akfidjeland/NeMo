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


/* The rest of the synapse data is only needed if querying synapses at
 * run-time. This data is stored separately */
struct AxonTerminalAux
{
	id32_t id;
	bool plastic;

	AxonTerminalAux(id32_t id, bool plastic) :
		id(id), plastic(plastic) { }
};


/* A row contains a number of synapses with a fixed source and delay. A
 * fixed-point format is used internally. The caller needs to specify the
 * format.  */
struct Row
{
	Row() : len(0) {}

	/* \post synapse order is the same as in input vector */
	Row(const std::vector<IdAxonTerminal>&, unsigned fbits);

	size_t len;
	boost::shared_array< FAxonTerminal<fix_t> > data;
};



struct SynapseAddress
{
	size_t row;
	sidx_t synapse;

	SynapseAddress(size_t row, sidx_t synapse):
		row(row), synapse(synapse) { }

	SynapseAddress():
		row(~0), synapse(~0) { }
};


class NetworkImpl;
class ConfigurationImpl;


/* Generic connectivity matrix
 *
 * Data in this class is organised for optimal cache performance. A
 * user-defined fixed-point format is used.
 */
class NEMO_BASE_DLL_PUBLIC ConnectivityMatrix
{
	public:

		typedef Mapper<nidx_t, nidx_t> mapper_t;

		//! \todo remove this ctor
		ConnectivityMatrix(const ConfigurationImpl& conf);

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
		Row& setRow(nidx_t source, delay_t,
				const std::vector<IdAxonTerminal>&,
				const mapper_t&);

		/*! \return all synapses for a given source and delay */
		const Row& getRow(nidx_t source, delay_t) const;

		/*! \return all synapses for a given source */
		void getSynapses(
				unsigned source,
				std::vector<unsigned>& targets,
				std::vector<unsigned>& delays,
				std::vector<float>& weights,
				std::vector<unsigned char>& plastic) const;

		const std::vector<float>& getWeights(const std::vector<synapse_id>& synapses);

		void finalize(const mapper_t&);

		typedef std::set<delay_t>::const_iterator delay_iterator;

		delay_iterator delay_begin(nidx_t source) const;
		delay_iterator delay_end(nidx_t source) const;

		unsigned fractionalBits() const { return m_fractionalBits; }

		delay_t maxDelay() const { return m_maxDelay; }

		void accumulateStdp(const std::vector<uint64_t>& recentFiring);

		void applyStdp(float reward);

	private:

		unsigned m_fractionalBits;

		/* During network construction we accumulate data in a map. This way we
		 * don't need to know the number of neurons or the number of delays in
		 * advance */
		typedef boost::tuple<nidx_t, delay_t> fidx;
		std::map<fidx, Row> m_acc;

		/* In order to be able to read back synapse data at run-time we record
		 * some data in a separate map. Weights need to be read from m_cm as
		 * they can change at run-time. */
		std::map<fidx, std::vector<unsigned char> > m_plastic;

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
		std::vector<float> m_queriedWeights;

		/* Per-neuron list of full synapse addresses. The index here is the
		 * 32-bit synapse id provided by the input network. */
		typedef std::vector<SynapseAddress> address_row;
		typedef std::map<nidx_t, address_row> address_map;

		address_map m_synapseAddresses;

		/* The synapse address map is constructed lazily (on a per-neuron
		 * basis) to avoid having to store this for the full CM when the user
		 * is not querying synapses */
		void updateSynapseAddressMap(nidx_t source);

		/*! \return the full synapse address of the given synapse. Cache the
		 * mapping as a side effect. */
		SynapseAddress synapseAddress(synapse_id);

		/* Additional synapse data which is only needed for runtime queries.
		 * This is kept separate from m_cm so that we can make m_cm fast and
		 * compact. The query information is not crucial for performance.  */

		typedef std::vector<AxonTerminalAux> aux_row;
		typedef std::map<fidx, aux_row> aux_map;
		aux_map m_cmAux;
};



inline
size_t
ConnectivityMatrix::addressOf(nidx_t source, delay_t delay) const
{
	return source * m_maxDelay + delay - 1;
}

} // end namespace nemo

#endif
