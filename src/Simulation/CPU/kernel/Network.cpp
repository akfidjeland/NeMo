#include "Network.hpp"

#include <cmath>

extern "C" {
#include "cpu_kernel.h"
}


Network::Network(
		double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[], //set to 0 if not thalamic input required
		unsigned int ncount,
		delay_t maxDelay) :
	cm(ncount, maxDelay),
	fired(ncount, 0),
	recentFiring(ncount, 0)
{
	//! \todo pre-allocate neuron data
	for(size_t i=0; i < ncount; ++i) {
		state.push_back(NState(u[i], v[i], sigma[i]));
		param.push_back(NParam(a[i], b[i], c[i], d[i]));
	}

	/* This RNG state vector needs to be filled with initialisation data. Each
	 * RNG needs 4 32-bit words of seed data. We use just a single RNG now, but
	 * should have one per thread for later so that we can get repeatable
	 * results.
	 *
	 * Fill it up from lrand48 -- in practice you would probably use something
	 * a bit better. */
	srand48(0);
	rng.resize(4);
	for(unsigned i=0; i<4; ++i) {
		rng[i] = ((unsigned) lrand48()) << 1;
	}
}



void
add_synapses(NETWORK net,
		nidx_t source,
		delay_t delay,
		nidx_t* targets,
		weight_t* weights,
		size_t length)
{
	net->cm.setRow(source, delay, targets, weights, length);
}
