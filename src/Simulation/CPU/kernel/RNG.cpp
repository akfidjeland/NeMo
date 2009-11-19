#include "RNG.hpp"
#include <cmath>
#include <cstdlib>


RNG::RNG() 
{
	/* This RNG state vector needs to be filled with initialisation data. Each
	 * RNG needs 4 32-bit words of seed data. We use just a single RNG now, but
	 * should have one per thread for later so that we can get repeatable
	 * results.
	 *
	 * Fill it up from lrand48 -- in practice you would probably use something
	 * a bit better. */
	srand48(0);
	for(unsigned i=0; i<4; ++i) {
		m_state[i] = ((unsigned) lrand48()) << 1;
	}
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
