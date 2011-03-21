#include "Neuron.hpp"

#include <boost/format.hpp>

#include "exception.hpp"

namespace nemo {


Neuron::Neuron(const NeuronType& type) :
	 a(0), b(0), c(0), d(0), u(0), v(0), sigma(0)
{
	init(type);
}


Neuron::Neuron(const NeuronType& type, float fParam[], float fState[])
{
	init(type);
	set(fParam, fState);
}



void
Neuron::init(const NeuronType& type)
{
	/* Currently this is hard-coded to use Izhikevich neurons only */
	if(type.f_nParam() != 5 && type.f_nState() != 2) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Unsupported neuron type");
	}
}



void
Neuron::set(float fParam[], float fState[])
{
	a = fParam[0];
	b = fParam[1];
	c = fParam[2];
	d = fParam[3];
	sigma = fParam[4];
	u = fState[0];
	v = fState[1];
}



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



void
Neuron::f_setParameter(size_t i, float val)
{
	using boost::format;
	switch(i) {
		case 0: a = val; break;
		case 1: b = val; break;
		case 2: c = val; break;
		case 3: d = val; break;
		case 4: sigma = val; break;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron parameter index (%u)") % i));
	}
}



void
Neuron::f_setState(size_t i, float val)
{
	using boost::format;

	switch(i) {
		case 0: u = val; break;
		case 1: v = val; break;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron state variable index (%u)") % i));
	}
}

}
