#ifndef NETWORK_HPP
#define NETWORK_HPP


#include <vector>
#include <stdint.h>

#include "ConnectivityMatrix.hpp"


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

		size_t m_neuronCount;

		delay_t m_maxDelay;

		uint m_cycle;

		void deliverSpikesOne(nidx_t source, delay_t delay);
};


#endif
