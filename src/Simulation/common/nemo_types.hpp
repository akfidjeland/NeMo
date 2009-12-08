#ifndef NEMO_TYPES_HPP
#define NEMO_TYPES_HPP

#include "nemo_types.h"

namespace nemo {

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
