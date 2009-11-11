#ifndef CONNECTIVITY_MATRIX_HPP
#define CONNECTIVITY_MATRIX_HPP

#include <vector>

#include "types.h"


struct Synapse
{
	Synapse(weight_t w, nidx_t t) : weight(w), target(t) {}

	weight_t weight; 
	nidx_t target; 
};


class ConnectivityMatrix
{
	public:

		ConnectivityMatrix(size_t neuronCount, size_t maxDelay);

		/*! Add synapses for a particular presynaptic neuron and a particular delay */
		void setRow(
				nidx_t source,
				delay_t delay,
				const nidx_t* targets,
				const weight_t* weights,
				size_t length);

	private:

		/* Synapses are stored per presynaptic and per delay */
		std::vector< std::vector<Synapse> > m_cm;

		size_t m_neuronCount;
		size_t m_maxDelay;

		/*! \return linear index into CM, based on 2D index (neuron,delay) */			
		size_t addressOf(nidx_t, delay_t); 
};

#endif
