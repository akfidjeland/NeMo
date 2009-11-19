#ifndef NETWORK_HPP
#define NETWORK_HPP

#ifdef PTHREADS_ENABLED
#include <pthread.h>
#endif

#include <vector>
#include <stdint.h>

#include "ConnectivityMatrix.hpp"
#include "Timer.hpp"

#ifdef PTHREADS_ENABLED
struct Job {
	size_t start;
	size_t end;
	unsigned int* fstim;
	struct Network* net;
};
#endif


struct Network {

	public:

		Network(fp_t a[],
			fp_t b[],
			fp_t c[],
			fp_t d[],
			fp_t u[],
			fp_t v[],
			fp_t sigma[], //set to 0 if not thalamic input required
			size_t ncount,
			delay_t maxDelay);

		~Network();

		/*! Add synapses for a particular presynaptic neuron and a particular
		 * delay */
		void setCMRow(nidx_t source, delay_t delay,
				const nidx_t* targets, const weight_t* weights, size_t length);

		/*! Deliver spikes and update neuron state */
		void step(unsigned int fstim[]);

		/*! Update state of all neurons */
		void update(unsigned int fstim[]);

		/*! Deliver spikes due for delivery */
		void deliverSpikes();

		const std::vector<unsigned int>& readFiring() const;

		/*! \return number of milliseconds of wall-clock time elapsed since first
		 * simulation step */
		long int elapsed();

		void resetTimer();

	private:

		//! \todo enforce 16-byte allignment to support vectorisation
		std::vector<fp_t> m_a;
		std::vector<fp_t> m_b;
		std::vector<fp_t> m_c;
		std::vector<fp_t> m_d;

		std::vector<fp_t> m_u;
		std::vector<fp_t> m_v;
		std::vector<fp_t> m_sigma;

		std::vector<unsigned int> m_pfired;

		ConnectivityMatrix m_cm;

		/* last 64 cycles worth of firing, one entry per neuron */
		std::vector<uint64_t> m_recentFiring;

		/* accumulated current from incoming spikes for each neuron */
		std::vector<fp_t> m_current;

		// may want to have one rng per neuron or at least per thread
		std::vector<unsigned int> m_rng; // fixed length: 4

		/* compacted firing for the last cycle's worth of firing, one entry per
		 * fired neuron */
		std::vector<unsigned int> m_fired;

#ifdef PTHREADS_ENABLED
		//! \todo allow user to determine number of threads
		static const int m_nthreads = 4;
		pthread_t m_thread[m_nthreads];
		pthread_attr_t m_thread_attr[m_nthreads];
		Job m_job[m_nthreads];

		void initThreads();

		friend void* start_thread(void*);
#endif

		size_t m_neuronCount;

		delay_t m_maxDelay;

		uint m_cycle;

		void deliverThalamic();
		void updateRange(int begin, int end, unsigned int* fstim);
		void processFired();

		void deliverSpikesOne(nidx_t source, delay_t delay);

		Timer m_timer;
};


#endif
