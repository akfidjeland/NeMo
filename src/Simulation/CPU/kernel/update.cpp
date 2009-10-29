extern "C" {
#include "cpu_kernel.h"
}

#include <vector>
#include <math.h>

#define SUBSTEPS 4
#define SUBSTEP_MULT 0.25

//#define VERBOSE

struct NParam {

	NParam(double a, double b, double c, double d) :
		a(a), b(b), c(c), d(d) {}

	double a;
	double b;
	double c;
	double d;
};


struct NState {

	NState(double u, double v, double sigma) :
		u(u), v(v), sigma(sigma) {}

	double u;
	double v;
	double sigma;
};


struct Network {
	std::vector<NParam> param;
	std::vector<NState> state;	

	// last cycle's worth of firing, one entry per neuron
	std::vector<bool_t> fired; 

	// may want to have one rng per neuron or at least per thread
	unsigned int rng[4];
};



struct Network*
set_network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[], //set to 0 if not thalamic input required
		unsigned int len)
{
	//! \todo pre-allocate neuron data
	Network* net = new Network();
	for(size_t i=0; i < len; ++i) {
		net->state.push_back(NState(u[i], v[i], sigma[i]));
		net->param.push_back(NParam(a[i], b[i], c[i], d[i]));
	}
	net->fired.resize(len, 0);

	/* This RNG state vector needs to be filled with initialisation data. Each
	 * RNG needs 4 32-bit words of seed data. We use just a single RNG now, but
	 * should have one per thread for later so that we can get repeatable
	 * results.
	 *
	 * Fill it up from lrand48 -- in practice you would probably use something
	 * a bit better. */
	//! \todo move this into ctor of network
	srand48(0);
	for(unsigned i=0; i<4; ++i) {
		net->rng[i] = ((unsigned) lrand48()) << 1;
	}
	return net;
}



void
delete_network(Network* net)
{
	delete net; 
}



unsigned
rng_genUniform(unsigned* rngState)
{
	unsigned t = (rngState[0]^(rngState[0]<<11));
	rngState[0] = rngState[1];
	rngState[1] = rngState[2];
	rngState[2] = rngState[3];
	rngState[3] = (rngState[3]^(rngState[3]>>19))^(t^(t>>8));
	return rngState[3];
}



/* For various reasons this generates a pair of samples for each call. If nesc.
 * then you can just stash one of them somewhere until the next time it is
 * needed or something.  */
float
rng_genGaussian(unsigned* rngState)
{
	float a = rng_genUniform(rngState) * 1.4629180792671596810513378043098e-9f;
	float b = rng_genUniform(rngState) * 0.00000000023283064365386962890625f;
	float r = sqrtf(-2*logf(1-b));
	// cosf(a) * r // ignore the second random
	return sinf(a) * r;
}



inline
bool_t
updateNeuron(const NParam& param,
		unsigned int stimulated,
		double I,
		NState& state,
		unsigned* rng)
{
	bool fired = false;

	double a = param.a;
	double b = param.b;
	double u = state.u;
	double v = state.v;

	/* thalamic input */
	if(state.sigma != 0.0f) {
		I += state.sigma * (double) rng_genGaussian(rng);
	}

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
			updateNeuron(network->param[n],
					fstim[n],
					current[n],
					network->state[n],
					network->rng);
#ifdef VERBOSE
		if(network->fired[n]) {
			fprintf(stderr, "n%u fired\n", n);
		}
#endif
	}
	return &network->fired[0];
}
