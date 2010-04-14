#ifndef NEMO_CONNECTIVITY_HPP
#define NEMO_CONNECTIVITY_HPP

#include <map>
#include <vector>
#include <boost/tuple/tuple.hpp>

#include "nemo_types.hpp"

namespace nemo {

	namespace cuda {
		// needed for 'friend' declarations
		class ConnectivityMatrix;
		class NeuronParameters;
	}


class Connectivity
{
	public :

		Connectivity();

		void addNeuron(nidx_t neuronIndex,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		/* Add a single synapse */
		void addSynapse(
				nidx_t source,
				delay_t delay,
				nidx_t target,
				weight_t weight,
				uchar isPlastic);

		//! \todo remove addSynapses from this interface
		/*! Add multiple synapses with the same source neuron */
		void addSynapses(
				nidx_t source,
				const std::vector<nidx_t>& targets,
				const std::vector<delay_t>& delays,
				const std::vector<weight_t>& weights,
				const std::vector<uchar> is_plastic);

		nidx_t maxSourceIdx() const { return m_maxSourceIdx; }
		delay_t maxDelay() const { return m_maxDelay; }
		weight_t maxWeight() const { return m_maxWeight; }
		weight_t minWeight() const { return m_minWeight; }

	private :

		//! \todo perhaps store this as double?
		typedef nemo::Neuron<float> neuron_t;
		std::map<nidx_t, neuron_t> m_neurons;

		typedef boost::tuple<nidx_t, weight_t, unsigned char> synapse_t;
		typedef std::vector<synapse_t> bundle_t;
		typedef std::map<delay_t, bundle_t> axon_t;
		typedef std::map<nidx_t, axon_t> fcm_t;

		fcm_t m_fcm;

		nidx_t m_maxSourceIdx;
		delay_t m_maxDelay;
		weight_t m_maxWeight;
		weight_t m_minWeight;

		friend class cuda::ConnectivityMatrix;
		friend class cuda::NeuronParameters;
};

} // end namespace nemo
#endif
