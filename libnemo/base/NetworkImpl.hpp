#ifndef NEMO_NETWORK_IMPL_HPP
#define NEMO_NETWORK_IMPL_HPP

//! \file NetworkImpl.hpp

#include <map>
#include <vector>

#include "nemo_config.h"
#include "types.hpp"

namespace nemo {

	namespace cuda {
		// needed for 'friend' declarations
		class ConnectivityMatrix;
		class NeuronParameters;
		class ThalamicInput;
	}

	namespace mpi {
		class Master;
	}


class NetworkImpl
{
	public :

		NetworkImpl();

		void addNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		void addNeuron(nidx_t nidx, const Neuron<float>&);

		void addSynapse(
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight,
				unsigned char plastic);

		void addSynapses(
				unsigned source,
				const std::vector<unsigned>& targets,
				const std::vector<unsigned>& delays,
				const std::vector<float>& weights,
				const std::vector<unsigned char>& plastic);

		/* lower-level interface using raw C arrays. This is mainly intended
		 * for use in foreign language interfaces such as C and Mex, where
		 * constructing std::vectors would be redundant. */
		template<typename N, typename D, typename W, typename B>
		void addSynapses(
				N source,
				const N targets[],
				const D delays[],
				const W weights[],
				const B plastic[],
				size_t len);

		nidx_t maxSourceIdx() const { return m_maxSourceIdx; }
		delay_t maxDelay() const { return m_maxDelay; }
		weight_t maxWeight() const { return m_maxWeight; }
		weight_t minWeight() const { return m_minWeight; }

		unsigned neuronCount() const;

	private :

		//! \todo perhaps store this as double?
		typedef nemo::Neuron<weight_t> neuron_t;
		std::map<nidx_t, neuron_t> m_neurons;

		typedef AxonTerminal<nidx_t, weight_t> synapse_t;
		typedef std::vector<synapse_t> bundle_t;
		//! \todo could keep this in a single map with a tuple index
		typedef std::map<delay_t, bundle_t> axon_t;
		typedef std::map<nidx_t, axon_t> fcm_t;

		fcm_t m_fcm;

		int m_maxSourceIdx;
		delay_t m_maxDelay;
		weight_t m_maxWeight;
		weight_t m_minWeight;

		friend class cuda::ConnectivityMatrix;
		friend class cuda::NeuronParameters;
		friend class cuda::ThalamicInput;
		friend class mpi::Master;
};


} // end namespace nemo
#endif
