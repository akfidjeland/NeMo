#ifndef NEMO_TYPES_HPP
#define NEMO_TYPES_HPP

#include "nemo_types.h"

namespace nemo {


template<typename FP>
struct Neuron {

	Neuron(): a(0), b(0), c(0), d(0), u(0), v(0), sigma(0) {}

	Neuron(FP a, FP b, FP c, FP d, FP u, FP v, FP sigma) :
		a(a), b(b), c(c), d(d), u(u), v(v), sigma(sigma) {}

	FP a, b, c, d, u, v, sigma;
};



//! \todo can probably get rid of this. Use boost::tuple instead
struct ForwardIdx
{
	ForwardIdx(nidx_t source, delay_t delay) :
		source(source), delay(delay) {}

	nidx_t source;
	delay_t delay;
};



inline
bool
operator<(const ForwardIdx& a, const ForwardIdx& b)
{
	return a.source < b.source || (a.source == b.source && a.delay < b.delay);
}

};

#endif
