#ifndef NEMO_NETWORK_GENERATOR_HPP
#define NEMO_NETWORK_GENERATOR_HPP

#include <nemo/types.hpp>
#include <nemo/network/iterator.hpp>


namespace nemo {
	namespace network {

/* A network generator is simply a class which can produce a sequence of
 * neurons and a sequence of synapses. Network generators are expected to
 * provide all neurons first, then all synapses. */
class Generator
{
	public : 
		
		virtual neuron_iterator neuron_begin() const = 0;
		virtual neuron_iterator neuron_end() const = 0;

		//! \todo add synapse iterators as well
		//virtual iterator<synapse> synapse_begin() const = 0;
		//virtual iterator<synapse> synapse_end() const = 0;
};


	} // end namespace network
} // end namespace nemo


#endif
