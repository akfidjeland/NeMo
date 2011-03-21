#include "Neuron.hpp"

#include <algorithm>
#include <boost/format.hpp>

#include "exception.hpp"

namespace nemo {


Neuron::Neuron(const NeuronType& type) :
	mf_param(type.f_nParam(), 0.0f),
	mf_state(type.f_nState(), 0.0f)
{ }


Neuron::Neuron(const NeuronType& type, float fParam[], float fState[]) :
	mf_param(fParam, fParam + type.f_nParam()),
	mf_state(fState, fState + type.f_nState())
{ }



void
Neuron::set(float fParam[], float fState[])
{
	std::copy(fParam, fParam + mf_param.size(), mf_param.begin());
	std::copy(fState, fState + mf_state.size(), mf_state.begin());
}



const float&
Neuron::f_paramRef(size_t i) const
{
	using boost::format;
	if(i >= mf_param.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron parameter index (%u)") % i));
	}
	return mf_param[i];
}




const float&
Neuron::f_stateRef(size_t i) const
{
	using boost::format;
	if(i >= mf_state.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron state variable index (%u)") % i));
	}
	return mf_state[i];
}



void
Neuron::f_setParameter(size_t i, float val)
{
	const_cast<float&>(f_paramRef(i)) = val;
}



void
Neuron::f_setState(size_t i, float val)
{
	const_cast<float&>(f_stateRef(i)) = val;
}

}
