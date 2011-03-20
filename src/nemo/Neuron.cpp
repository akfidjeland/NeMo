#include "Neuron.hpp"

#include <boost/format.hpp>

#include "exception.hpp"

namespace nemo {



float
Neuron::f_getParameter(size_t i) const
{
	using boost::format;

	switch(i) {
		case 0: return a;
		case 1: return b;
		case 2: return c;
		case 3: return d;
		case 4: return sigma;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron parameter index (%u)") % i));
	}
}




float
Neuron::f_getState(size_t i) const
{
	using boost::format;

	switch(i) {
		case 0: return u;
		case 1: return v;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron state variable index (%u)") % i));
	}
}


}
