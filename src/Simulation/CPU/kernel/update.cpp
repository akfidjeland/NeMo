extern "C" {
#include "cpu_kernel.h"
}

#include <vector>

#define SUBSTEPS 4
#define SUBSTEP_MULT 0.25

struct NParam {

	NParam(double a, double b, double c, double d) :
		a(a), b(b), c(c), d(d) {}

	double a;
	double b;
	double c;
	double d;
};


struct NState {

	NState(double u, double v) :
		u(u), v(v) {}

	double u;
	double v;
};


struct Network {
	std::vector<NParam> param;
	std::vector<NState> state;	

	// last cycle's worth of firing, one entry per neuron
	std::vector<bool_t> fired; 
};



struct Network*
set_network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		unsigned int len)
{
	//! \todo pre-allocate neuron data
	Network* net = new Network();
	for(size_t i=0; i < len; ++i) {
		net->state.push_back(NState(u[i], v[i]));
		net->param.push_back(NParam(a[i], b[i], c[i], d[i]));
	}
	net->fired.resize(len, 0);
	return net;
}



void
delete_network(Network* net)
{
	delete net; 
}



inline
bool_t
updateNeuron(const NParam& param,
		unsigned int stimulated,
		double I,
		NState& state)
{
	bool fired = false;

	double a = param.a;
	double b = param.b;
	double u = state.u;
	double v = state.v;

	//! \todo explicitly unroll
	for(unsigned int t=0; t<SUBSTEPS; ++t) {
		//! \todo just exit from loop if fired
		if(!fired) { 
			v += SUBSTEP_MULT * ((0.04*v + 5.0) * v + 140.0 - u + I);
			/*! \todo: could pre-multiply this with a, when initialising memory */
			u += SUBSTEP_MULT * (a * (b*v - u));
			fired = v >= 30.0;
		} 
	}

	fired |= stimulated;

	if(fired) {
		v = param.c;
		u += param.d;
	}

	state.u = u;
	state.v = v;

	return fired ? 1 : 0;
}



bool_t*
update(Network* network, unsigned int fstim[], double current[])
{
	//! \todo update in parallel?
	for(size_t n=0; n < network->param.size(); ++n) {
		network->fired[n] = 
			updateNeuron(network->param[n], fstim[n], current[n], network->state[n]);
	}
	return &network->fired[0];
}
