#ifndef NETWORK_HPP
#define NETWORK_HPP


#include <vector>
#include <stdint.h>

#include "ConnectivityMatrix.hpp"



struct NParam {

	NParam(fp_t a, fp_t b, fp_t c, fp_t d) :
		a(a), b(b), c(c), d(d) {}

	fp_t a;
	fp_t b;
	fp_t c;
	fp_t d;
};


struct NState {

	NState(fp_t u, fp_t v, fp_t sigma) :
		u(u), v(v), sigma(sigma) {}

	fp_t u;
	fp_t v;
	fp_t sigma;
};



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

		/*! Add synapses for a particular presynaptic neuron and a particular delay */
		void setCMRow(nidx_t source, delay_t delay,
				const nidx_t* targets, const weight_t* weights, size_t length);

		/*! Deliver spikes and update neuron state */
		bool_t* step(unsigned int fstim[]);

		/*! Update state of all neurons */
		bool_t* update(unsigned int fstim[]);

		/*! Deliver spikes due for delivery */
		const std::vector<fp_t>& deliverSpikes();

	private:

		std::vector<NParam> m_param;
		std::vector<NState> m_state;
		ConnectivityMatrix m_cm;

		/* last 64 cycles worth of firing, one entry per neuron */
		std::vector<uint64_t> m_recentFiring;

		/* accumulated current from incoming spikes for each neuron */
		std::vector<fp_t> m_current;

		// may want to have one rng per neuron or at least per thread
		std::vector<unsigned int> m_rng; // fixed length: 4

		/* last cycle's worth of firing, one entry per neuron */
		std::vector<bool_t> m_fired;

		size_t m_neuronCount;

		delay_t m_maxDelay;

		uint m_cycle;

		void deliverSpikesOne(nidx_t source, delay_t delay);
};


#endif
