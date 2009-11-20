#include <cmath>
#include <algorithm>
#ifdef PTHREADS_ENABLED
#include <sched.h> // for setting thread affinity
#endif

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
	m_cm(ncount),
	m_recentFiring(ncount, 0),
	m_current(ncount, 0),
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

#ifdef PTHREADS_ENABLED
	initThreads();
#endif
}


Network::~Network()
{
#ifdef PTHREADS_ENABLED
	for(int i=0; i<m_nthreads; ++i) {
		pthread_attr_destroy(&m_thread_attr[i]);
	}
#endif
}


#ifdef PTHREADS_ENABLED

/* Initialise threads, but do not start them. Set the attributes, esp. thread
 * affinity, and pre-allocate work */
void
Network::initThreads()
{
	int nthreads = m_nthreads;
	for(int i=0; i<nthreads; ++i) {
		pthread_attr_init(&m_thread_attr[i]);
		pthread_attr_setdetachstate(&m_thread_attr[i], PTHREAD_CREATE_JOINABLE);
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(i, &cpuset);
		pthread_attr_setaffinity_np(&m_thread_attr[i], sizeof(cpu_set_t), &cpuset);

		size_t jobSize = m_neuronCount/nthreads;
		size_t start = i * jobSize;
		//! \todo deal with special cases for small number of neurons
		size_t end = std::min((i+1) * jobSize, m_neuronCount);
		m_job[i] = new Job(start, end, m_neuronCount, this);
	}
}

#endif




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
Network::updateRange(int start, int end, const unsigned int fstim[], RNG* rng)
{
	for(int n=start; n < end; n++) {

		if(m_sigma[n] != 0.0f) {
			m_current[n] += m_sigma[n] * (fp_t) rng->gaussian();
		}

		m_pfired[n] = 0;

		for(unsigned int t=0; t<SUBSTEPS; ++t) {
			if(!m_pfired[n]) {
				m_v[n] += SUBSTEP_MULT * ((0.04* m_v[n] + 5.0) * m_v[n] + 140.0 - m_u[n] + m_current[n]);
				/*! \todo: could pre-multiply this with a, when initialising memory */
				m_u[n] += SUBSTEP_MULT * (m_a[n] * (m_b[n] * m_v[n] - m_u[n]));
				m_pfired[n] = m_v[n] >= 30.0;
			}
		}

		m_pfired[n] |= fstim[n];
		m_recentFiring[n] = (m_recentFiring[n] << 1) | (uint64_t) m_pfired[n];

		if(m_pfired[n]) {
			m_v[n] = m_c[n];
			m_u[n] += m_d[n];
#ifdef DEBUG_TRACE
			fprintf(stderr, "c%u: n%u fired\n", m_cycle, n);
#endif
		}

	}
}



#ifdef PTHREADS_ENABLED

void*
start_thread(void* job_in)
{
	Job* job = static_cast<Job*>(job_in);
	(job->net)->updateRange(job->start, job->end, job->fstim, &job->rng);
	pthread_exit(NULL);
}

#endif



void
Network::update(unsigned int fstim[])
{
#ifdef PTHREADS_ENABLED
	for(int i=0; i<m_nthreads; ++i) {
		m_job[i]->fstim = &fstim[0];
		pthread_create(&m_thread[i], &m_thread_attr[i], start_thread, (void*) m_job[i]);
	}
	for(int i=0; i<m_nthreads; ++i) {
		pthread_join(m_thread[i], NULL);
	}
#else
	updateRange(0, m_neuronCount, fstim, &m_rng);
#endif

	m_cycle++;
}



const std::vector<unsigned int>&
Network::readFiring()
{
	m_fired.clear();
	for(int n=0; n < m_neuronCount; ++n) {
		if(m_pfired[n]) {
			m_fired.push_back(n);
		}
	}
	return m_fired;
}



void
Network::deliverSpikes()
{
	//! \todo reset the timer somewhere more sensible
	if(m_cycle == 0) {
		resetTimer();
		m_cm.finalize();
	}

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
	const Row& row = m_cm.getRow(source, delay);
	const Synapse* ss = row.data;

	/* could use pointer-arithmetic here, but profiling seems to indicate that
	 * it gives no advantage */
	for(int i=0; i < row.len; ++i) {
		const Synapse& s = ss[i];
		m_current[s.target] += s.weight;
#ifdef DEBUG_TRACE
		fprintf(stderr, "c%u: n%u -> n%u: %+f (delay %u)\n",
				m_cycle, source, s.target, s.weight, delay);
#endif
	}
}



long
Network::elapsed()
{
	return m_timer.elapsed();
}



void
Network::resetTimer()
{
	m_timer.reset();
}
