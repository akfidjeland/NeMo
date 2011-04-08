#ifndef NEMO_NETWORK_GENERATOR_HPP
#define NEMO_NETWORK_GENERATOR_HPP

#include <nemo/types.hpp>
#include <nemo/network/iterator.hpp>
#include <nemo/NeuronType.hpp>


namespace nemo {
	namespace network {

/* A network generator is simply a class which can produce a sequence of
 * neurons and a sequence of synapses. Network generators are expected to
 * provide all neurons first, then all synapses. */
class Generator
{
	public : 

		virtual ~Generator() { }

		typedef std::pair<nidx_t, Neuron> neuron;
		typedef Synapse synapse;
		
		virtual neuron_iterator neuron_begin() const = 0;
		virtual neuron_iterator neuron_end() const = 0;

		virtual synapse_iterator synapse_begin() const = 0;
		virtual synapse_iterator synapse_end() const = 0;

		virtual unsigned neuronCount() const = 0;

		virtual unsigned minNeuronIndex() const = 0;
		virtual unsigned maxNeuronIndex() const = 0;

		/*! \return the \i unique neuron type found in this network */
		virtual const class NeuronType& neuronType() const = 0;
};


	} // end namespace network
} // end namespace nemo


#endif
