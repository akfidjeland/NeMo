#include <cmath>

#include "Network.hpp"


#define SUBSTEPS 4
#define SUBSTEP_MULT 0.25


Network::Network(
		fp_t a[],
		fp_t b[],
		fp_t c[],
		fp_t d[],
		fp_t u[],
		fp_t v[],
		fp_t sigma[], //set to 0 if thalamic input not required
		size_t ncount,
		delay_t maxDelay) :
	m_a(ncount),
	m_b(ncount),
	m_c(ncount),
	m_d(ncount),
	m_u(ncount),
	m_v(ncount),
	m_sigma(ncount),
	m_pfired(ncount, 0),
	m_cm(ncount, maxDelay),
	m_recentFiring(ncount, 0),
	m_current(ncount, 0),
	m_rng(4, 0),
	m_neuronCount(ncount),
	m_maxDelay(maxDelay),
	m_cycle(0)
{
	std::copy(a, a + ncount, m_a.begin());
	std::copy(b, b + ncount, m_b.begin());
	std::copy(c, c + ncount, m_c.begin());
	std::copy(d, d + ncount, m_d.begin());
	std::copy(u, u + ncount, m_u.begin());
	std::copy(v, v + ncount, m_v.begin());
	std::copy(sigma, sigma + ncount, m_sigma.begin());

	/* This RNG state vector needs to be filled with initialisation data. Each
	 * RNG needs 4 32-bit words of seed data. We use just a single RNG now, but
	 * should have one per thread for later so that we can get repeatable
	 * results.
	 *
	 * Fill it up from lrand48 -- in practice you would probably use something
	 * a bit better. */
	srand48(0);
	for(unsigned i=0; i<4; ++i) {
		m_rng[i] = ((unsigned) lrand48()) << 1;
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



void
Network::setCMRow(nidx_t source, delay_t delay,
			const nidx_t* targets, const weight_t* weights, size_t length)
{
	m_cm.setRow(source, delay, targets, weights, length);
}



void
Network::step(unsigned int fstim[])
{
	deliverSpikes();
	update(fstim);
}



void
Network::update(unsigned int fstim[])
{
	m_fired.clear();

	for(int n=0; n < m_neuronCount; n+=1) {

		m_pfired[n] = 0;

		/* thalamic input */
		if(m_sigma[n] != 0.0f) {
			m_current[n] += m_sigma[n] * (fp_t) rng_genGaussian(&m_rng[0]);
		}

		for(unsigned int t=0; t<SUBSTEPS; ++t) {
			if(!m_pfired[n]) {
				m_v[n] += SUBSTEP_MULT * ((0.04* m_v[n] + 5.0) * m_v[n] + 140.0 - m_u[n] + m_current[n]);
				/*! \todo: could pre-multiply this with a, when initialising memory */
				m_u[n] += SUBSTEP_MULT * (m_a[n] * (m_b[n] * m_v[n] - m_u[n]));
				m_pfired[n] = m_v[n] >= 30.0;
			}
		}

		m_pfired[n] |= fstim[n];

		if(m_pfired[n]) {
			m_v[n] = m_c[n];
			m_u[n] += m_d[n];
			m_fired.push_back(n);
#ifdef DEBUG_TRACE
			fprintf(stderr, "c%u: n%u fired\n", m_cycle, n);
#endif
		}

		m_recentFiring[n] = (m_recentFiring[n] << 1) | (uint64_t) m_pfired[n];
	}

	m_cycle++;
}



const std::vector<unsigned int>&
Network::readFiring() const
{
	return m_fired;
}



void
Network::deliverSpikes()
{
	/* Ignore spikes outside of max delay. We keep these older spikes as they
	 * may be needed for STDP */
	uint64_t validSpikes = ~(((uint64_t) (~0)) << m_maxDelay);

	std::fill(m_current.begin(), m_current.end(), 0);

	for(size_t source=0; source < m_neuronCount; ++source) {

		//! \todo make use of delay bits here to avoid looping
		uint64_t f = m_recentFiring[source] & validSpikes;

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
}


void
Network::deliverSpikesOne(nidx_t source, delay_t delay)
{
	const std::vector<Synapse>& ss = m_cm.getRow(source, delay);

	for(std::vector<Synapse>::const_iterator s = ss.begin();
			s != ss.end(); ++s) {
		m_current[s->target] += s->weight;
#ifdef DEBUG_TRACE
		fprintf(stderr, "c%u: n%u -> n%u: %+f (delay %u)\n",
				m_cycle, source, s->target, s->weight, delay);
#endif
	}
}
