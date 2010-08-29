#ifndef NEMO_NETWORK_IMPL_HPP
#define NEMO_NETWORK_IMPL_HPP

//! \file NetworkImpl.hpp

#include <map>
#include <vector>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>

namespace nemo {


	namespace cuda {
		// needed for 'friend' declarations
		class ConnectivityMatrix;
		class NeuronParameters;
		class ThalamicInput;
	}

	namespace cpu {
		class Simulation;
	}

	class ConnectivityMatrix;

	namespace mpi {
		class Master;
	}

	namespace network {

		namespace programmatic {
			class synapse_iterator;
		}

class NEMO_BASE_DLL_PUBLIC NetworkImpl : public Generator
{
	public :

		NetworkImpl();

		void addNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		void addNeuron(nidx_t nidx, const Neuron<float>&);

		synapse_id addSynapse(
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight,
				unsigned char plastic);

		void addSynapses(
				const std::vector<unsigned>& sources,
				const std::vector<unsigned>& targets,
				const std::vector<unsigned>& delays,
				const std::vector<float>& weights,
				const std::vector<unsigned char>& plastic);

		/* lower-level interface using raw C arrays. This is mainly intended
		 * for use in foreign language interfaces such as C and Mex, where
		 * constructing std::vectors would be redundant. */
		template<typename N, typename D, typename W, typename B>
		void addSynapses(
				const N source[],
				const N targets[],
				const D delays[],
				const W weights[],
				const B plastic[],
				size_t len);

		void getSynapses(
				unsigned source,
				std::vector<unsigned>& targets,
				std::vector<unsigned>& delays,
				std::vector<float>& weights,
				std::vector<unsigned char>& plastic) const;

		/* pre: network is not empty */
		nidx_t minNeuronIndex() const;

		/* pre: network is not empty */
		nidx_t maxNeuronIndex() const;

		delay_t maxDelay() const { return m_maxDelay; }
		weight_t maxWeight() const { return m_maxWeight; }
		weight_t minWeight() const { return m_minWeight; }

		unsigned neuronCount() const;

		neuron_iterator neuron_begin() const;
		neuron_iterator neuron_end() const;

		synapse_iterator synapse_begin() const;
		synapse_iterator synapse_end() const;

	private :

		typedef nemo::Neuron<weight_t> neuron_t;
		std::map<nidx_t, neuron_t> m_neurons;

		typedef AxonTerminal synapse_t;
		typedef std::vector<synapse_t> bundle_t;
		//! \todo could keep this in a single map with a tuple index
		typedef std::map<delay_t, bundle_t> axon_t;
		typedef std::map<nidx_t, axon_t> fcm_t;

		fcm_t m_fcm;

		int m_minIdx;
		int m_maxIdx;
		delay_t m_maxDelay;
		weight_t m_minWeight;
		weight_t m_maxWeight;

		/* Keep track of the number of synapses per neuron in order to generate
		 * a dense list of synapse ids */
		std::map<nidx_t, id32_t> m_synapseCount;

		//! \todo modify public interface to avoid friendship here
		friend class nemo::cuda::ConnectivityMatrix;
		friend class nemo::cuda::NeuronParameters;
		friend class nemo::cuda::ThalamicInput;
		friend class nemo::ConnectivityMatrix;
		friend class nemo::cpu::Simulation;
		friend class nemo::mpi::Master;

		friend class programmatic::synapse_iterator;
};

	} // end namespace network
} // end namespace nemo
#endif
