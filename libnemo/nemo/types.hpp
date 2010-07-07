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
#include <nemo/nemo_config.h>
#include "types.h"

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




template<typename I, typename W>
class AxonTerminal
{
	public :

		I target;
		W weight;
		//! \todo change to bool?
		unsigned char plastic;

		AxonTerminal() : target(0), weight(0.0f), plastic(false) { }

		AxonTerminal(I t, W w, unsigned char p) : target(t), weight(w), plastic(p) {}

	private :
#ifdef INCLUDE_MPI
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & target;
			ar & weight;
			ar & plastic;
		}
#endif
};



template<typename I, typename D, typename W>
class Synapse
{
	public :

		Synapse() : source(0), delay(0) {}

		Synapse(I source, D delay, const AxonTerminal<I, W>& terminal) :
			source(source), delay(delay), terminal(terminal) { }

		I source;
		D delay;
		AxonTerminal<I, W> terminal;

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

} // end namespace nemo

#endif
