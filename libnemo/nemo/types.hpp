#ifndef NEMO_TYPES_HPP
#define NEMO_TYPES_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/* The basic types in nemo_types are also used without an enclosing namespace
 * inside the kernel (which is pure C). */
#include <nemo/config.h>
#include "internal_types.h"

#ifdef INCLUDE_MPI
#include <boost/serialization/serialization.hpp>
#endif


#ifdef INCLUDE_MPI
namespace boost {
	namespace serialization {
		class access;
	}
}
#endif

namespace nemo {


template<typename FP>
class Neuron
{
	public :

		Neuron(): a(0), b(0), c(0), d(0), u(0), v(0), sigma(0) {}

		Neuron(FP a, FP b, FP c, FP d, FP u, FP v, FP sigma) :
			a(a), b(b), c(c), d(d), u(u), v(v), sigma(sigma) {}

		FP a, b, c, d, u, v, sigma;

	private :

#ifdef INCLUDE_MPI
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & a;
			ar & b;
			ar & c;
			ar & d;
			ar & u;
			ar & v;
			ar & sigma;
		}
#endif

};



struct AxonTerminal
{
	public :

		id32_t id;
		nidx_t target;
		float weight;
		bool plastic;

		AxonTerminal():
			id(~0), target(~0), weight(0.0f), plastic(false) { }

		AxonTerminal(id32_t id, nidx_t t, float w, bool p):
			id(id), target(t), weight(w), plastic(p) { }

	private :
#ifdef INCLUDE_MPI
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & id;
			ar & target;
			ar & weight;
			ar & plastic;
		}
#endif
};



class Synapse
{
	public :

		Synapse() : source(0), delay(0) {}

		Synapse(nidx_t source, delay_t delay, const AxonTerminal& terminal) :
			source(source), delay(delay), terminal(terminal) { }

		nidx_t source;
		delay_t delay;
		AxonTerminal terminal;

		id32_t id() const { return terminal.id; }

		nidx_t target() const { return terminal.target; }

		unsigned char plastic() const { return terminal.plastic; }

		float weight() const { return terminal.weight; }

	private :

#ifdef INCLUDE_MPI
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & source;
			ar & delay;
			ar & terminal;
		}
#endif
};


class RSynapse
{
	public :

		RSynapse() :
			source(~0), delay(0), synapse(~0), w_diff(0) {}

		RSynapse(nidx_t source, delay_t delay, sidx_t fsynapse) :
			source(source), delay(delay), synapse(fsynapse), w_diff(0) { }

		nidx_t source;
		delay_t delay;
		sidx_t synapse; // index in the forward connectivity matrix
		fix_t w_diff;
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


} // end namespace nemo

#endif
