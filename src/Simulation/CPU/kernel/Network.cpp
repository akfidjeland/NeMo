#include "Network.hpp"


#include <cmath>

extern "C" {
#include "cpu_kernel.h"
}

#define SUBSTEPS 4
#define SUBSTEP_MULT 0.25


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
	recentFiring(ncount, 0),
	current(ncount, 0),
	m_fired(ncount, 0),
	m_neuronCount(ncount),
	m_maxDelay(maxDelay),
	m_cycle(0)
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



//! \todo move into NState class
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
	//! \todo put SUBSTEPS into static
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
Network::step(unsigned int fstim[])
{
	//! \todo use a separate step function instead
	const std::vector<fp_t>& current = deliverSpikes();

	//! \todo update in parallel?
	//! \todo factor out the update
	for(size_t n=0; n < param.size(); ++n) {
		bool fired = updateNeuron(param[n],
					fstim[n],
					current[n],
					state[n],
					&rng[0]);
		m_fired[n] = fired;
		recentFiring[n] = (recentFiring[n] << 1) | (fired ? 0x1 : 0x0);

#ifdef DEBUG_TRACE
		if(fired) {
			fprintf(stderr, "c%u: n%u fired\n", m_cycle, n);
		}
#endif
	}

	m_cycle++;

	return &m_fired[0];

}


const std::vector<fp_t>&
Network::deliverSpikes()
{
	/* Ignore spikes outside of max delay. We keep these older spikes as they
	 * may be needed for STDP */
	uint64_t validSpikes = ~(((uint64_t) (~0)) << m_maxDelay);

	std::fill(current.begin(), current.end(), 0);

	for(size_t source=0; source < m_neuronCount; ++source) {

		//! \todo make use of delay bits here to avoid looping
		uint64_t f = recentFiring[source] & validSpikes;

		//! \todo add sanity check to make sure that ffsll takes 64-bit
		int delay = 0;
		while(f) {
			//! \todo do this in a way that's 64-bit safe.
			int shift = ffsll(f);
			delay += shift;
			f = f >> shift;
			deliverSpikesOne(source, delay);
		}
	}

	return current;
}


void
Network::deliverSpikesOne(
		nidx_t source, delay_t delay)
{
	const std::vector<Synapse>& ss = cm.getRow(source, delay);

	for(std::vector<Synapse>::const_iterator s = ss.begin();
			s != ss.end(); ++s) {
		current[s->target] += s->weight;
#ifdef DEBUG_TRACE
		fprintf(stderr, "c%u: n%u -> n%u: %+f (delay %u)\n",
				m_cycle, source, s->target, s->weight, delay);
#endif
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
