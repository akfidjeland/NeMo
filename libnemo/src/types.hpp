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
#include "types.h"


namespace nemo {


template<typename FP>
struct Neuron {

	Neuron(): a(0), b(0), c(0), d(0), u(0), v(0), sigma(0) {}

	Neuron(FP a, FP b, FP c, FP d, FP u, FP v, FP sigma) :
		a(a), b(b), c(c), d(d), u(u), v(v), sigma(sigma) {}

	FP a, b, c, d, u, v, sigma;
};



template<typename I, typename W>
struct Synapse
{
	I target;
	W weight;
	//! \todo change to bool?
	unsigned char plastic;

	Synapse(I t, W w, unsigned char p) : target(t), weight(w), plastic(p) {}
};




};

#endif
