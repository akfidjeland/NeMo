#include "RNG.hpp"

#include <cmath>
#include <algorithm>
#include <boost/random.hpp>

namespace nemo {


RNG::RNG()
{
	std::fill(m_state, m_state+4, 0.0f);
}


RNG::RNG(unsigned seeds[])
{
	std::copy(seeds, seeds+4, m_state);
}



unsigned
RNG::uniform()
{
	unsigned t = (m_state[0]^(m_state[0]<<11));
	m_state[0] = m_state[1];
	m_state[1] = m_state[2];
	m_state[2] = m_state[3];
	m_state[3] = (m_state[3]^(m_state[3]>>19))^(t^(t>>8));
	return m_state[3];
}



/* For various reasons this generates a pair of samples for each call. If nesc.
 * then you can just stash one of them somewhere until the next time it is
 * needed or something.  */
float
RNG::gaussian()
{
	float a = uniform() * 1.4629180792671596810513378043098e-9f;
	float b = uniform() * 0.00000000023283064365386962890625f;
	float r = sqrtf(-2*logf(1-b));
	// cosf(a) * r // ignore the second random
	return sinf(a) * r;
}



void
initialiseRng(nidx_t minNeuronIdx, nidx_t maxNeuronIdx, std::vector<RNG>& rngs)
{
	//! \todo allow users to seed this RNG
	typedef boost::mt19937 rng_t;
	rng_t rng;

	boost::variate_generator<rng_t, boost::uniform_int<unsigned long> >
		seed(rng, boost::uniform_int<unsigned long>(0, 0x7fffffff));

	for(unsigned gidx=0; gidx < 4 * minNeuronIdx; ++gidx) {
		seed();
	}

	for(unsigned gidx = minNeuronIdx, gidx_end = maxNeuronIdx;
			gidx <= gidx_end; ++gidx) {
		unsigned seeds[4];
		for(unsigned plane=0; plane < 4; ++plane) {
			seeds[plane] = seed();
		}
		rngs[gidx] = RNG(seeds); // some of these neuron indices may be invalid
	}
}


} // end namespace
