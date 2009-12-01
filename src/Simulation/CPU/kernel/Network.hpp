#ifndef NETWORK_HPP
#define NETWORK_HPP

#ifdef PTHREADS_ENABLED
#include <pthread.h>
#endif

#include <vector>
#include <stdint.h>

#include "ConnectivityMatrix.hpp"
#include "RNG.hpp"
#include "common.h"

#include <Timer.hpp>
#include <STDP.hpp>

namespace nemo {
	namespace cpu {

#ifdef PTHREADS_ENABLED


struct Job {

	Job(): start(0), end(0), fstim(NULL), net(NULL) {}

	Job(size_t start, size_t end, size_t ncount, struct Network* net) :
		start(start), end(end), fstim(NULL), net(net) {
	}

	size_t start;
	size_t end;

	// input - full vector
	unsigned int* fstim;

	struct Network* net;

	RNG rng;

} __attribute((aligned(ASSUMED_CACHE_LINE_SIZE)));

#endif


struct Neuron {

	Neuron(): a(0), b(0), c(0), d(0), u(0), v(0), sigma(0) {}

	Neuron(fp_t a, fp_t b, fp_t c, fp_t d, fp_t u, fp_t v, fp_t sigma) :
		a(a), b(b), c(c), d(d), u(u), v(v), sigma(sigma) {}

	fp_t a, b, c, d, u, v, sigma;
};


struct Network {

	public:

		/*! Create an empty network which can be added to incrementally */
		Network();

		Network(fp_t a[],
			fp_t b[],
			fp_t c[],
			fp_t d[],
			fp_t u[],
			fp_t v[],
			fp_t sigma[], //set to 0 if not thalamic input required
			size_t ncount);

		~Network();

		/*! Add a single neuron */
		void addNeuron(nidx_t neuronIndex,
				fp_t a, fp_t b, fp_t c, fp_t d,
				fp_t u, fp_t v, fp_t sigma);

		/*! Add synapses for a particular presynaptic neuron and a particular
		 * delay */
		void addSynapses(nidx_t source, delay_t delay,
				const nidx_t targets[], const weight_t weights[],
				const uint plastic[], size_t length);

		/*! Set up runtime data structures after network construction is complete */
		void startSimulation();

		/*! Deliver spikes and update neuron state */
		void step(unsigned int fstim[]);

		/*! Update state of all neurons */
		void update(unsigned int fstim[]);

		/*! Deliver spikes due for delivery */
		void deliverSpikes();

		const std::vector<unsigned int>& readFiring();

		void applyStdp(double reward);

		/*! \return number of milliseconds of wall-clock time elapsed since first
		 * simulation step */
		long int elapsed();

		void resetTimer();

		size_t neuronCount() const { return m_neuronCount; }

		void configureStdp(const STDP<double>& conf);

	private:

		/* The network is constructed incrementally. */
		std::map<nidx_t, Neuron> m_acc;
		bool m_constructing;

		/* When all neurons are added, calling finalize will move data into
		 * runtime data structures */
		void finalize();

		void allocateNeuronData(size_t neuronCount);

		/* At run-time data is put into linear matrices for vectorizable
		 * operations */
		//! \todo enforce 16-byte allignment to support vectorisation
		std::vector<fp_t> m_a;
		std::vector<fp_t> m_b;
		std::vector<fp_t> m_c;
		std::vector<fp_t> m_d;

		std::vector<fp_t> m_u;
		std::vector<fp_t> m_v;
		std::vector<fp_t> m_sigma;

		void allocateRuntimeData(size_t neuronCount);

		std::vector<unsigned int> m_pfired;

		nemo::cpu::ConnectivityMatrix m_cm;

		/* last 64 cycles worth of firing, one entry per neuron */
		std::vector<uint64_t> m_recentFiring;

		/* accumulated current from incoming spikes for each neuron */
		std::vector<fp_t> m_current;

		// may want to have one rng per neuron or at least per thread
		RNG m_rng;

		/* compacted firing for the last cycle's worth of firing, one entry per
		 * fired neuron */
		std::vector<unsigned int> m_fired;

#ifdef PTHREADS_ENABLED
		//! \todo allow user to determine number of threads
		static const int m_nthreads = 4;
		pthread_t m_thread[m_nthreads];
		pthread_attr_t m_thread_attr[m_nthreads];
		Job* m_job[m_nthreads];

		void initThreads();

		friend void* start_thread(void*);
#endif

		size_t m_neuronCount;

		uint m_cycle;

		void updateRange(int begin, int end, const unsigned int fstim[], RNG* rng);

		void deliverSpikesOne(nidx_t source, delay_t delay);

		void accumulateStdp();

		Timer m_timer;

		STDP<double> m_stdp;

		weight_t updateRegion(uint64_t spikes, nidx_t source, nidx_t target);
};


	} // namespace cpu
} // namespace nemo
#endif
