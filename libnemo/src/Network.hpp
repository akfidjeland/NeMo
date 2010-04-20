#ifndef NEMO_CONNECTIVITY_HPP
#define NEMO_CONNECTIVITY_HPP

//! \file Network.hpp

#include <map>
#include <vector>

#include "types.hpp"

namespace nemo {

	namespace cuda {
		// needed for 'friend' declarations
		class ConnectivityMatrix;
		class NeuronParameters;
	}


/*! Networks are constructed by adding individual neurons, and single or groups
 * of synapses to the network. Neurons are given indices (from 0) which should
 * be unique for each neuron. When adding synapses the source or target neurons
 * need not necessarily exist yet, but should be defined before the network is
 * finalised. */
class Network
{
	public :

		Network();

		/*! Add a single neuron to the network
		 *
		 * The neuron uses the Izhikevich neuron model. See E. M. Izhikevich
		 * "Simple model of spiking neurons", \e IEEE \e Trans. \e Neural \e
		 * Networks, vol 14, pp 1569-1572, 2003 for a full description of the
		 * model and the parameters.
		 *
		 * \param idx
		 * 		Neuron index. This should be unique
		 * \param a
		 * 		Time scale of the recovery variable \a u
		 * \param b
		 * 		Sensitivity to sub-threshold fluctutations in the membrane
		 * 		potential \a v
		 * \param c
		 * 		After-spike reset value of the membrane potential \a v
		 * \param d
		 * 		After-spike reset of the recovery variable \a u
		 * \param u
		 * 		Initial value for the membrane recovery variable
		 * \param v
		 * 		Initial value for the membrane potential
		 * \param sigma
		 * 		Parameter for a random gaussian per-neuron process which
		 * 		generates random input current drawn from an N(0,\a sigma)
		 * 		distribution. If set to zero no random input current will be
		 * 		generated.
		 */
		void addNeuron(unsigned neuronIndex,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		/* Add a single synapse */
		void addSynapse(
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight,
				unsigned char plastic);

		//! \todo remove addSynapses from this interface use addSynape only
		//! \todo change from uchar to bool
		/*! Add to the network a group of synapses with the same presynaptic neuron
		 *
		 * \param source
		 * 		Index of source neuron
		 * \param targets
		 * 		Indices of target neurons
		 * \param delays
		 * 		Synapse conductance delays in milliseconds
		 * \param weights
		 * 		Synapse weights
		 * \param plastic
		 * 		Specifies for each synapse whether or not it is plastic.
		 * 		See section on STDP.
		 *
		 * \pre
		 * 		\a targets, \a delays, \a weights, and \a plastic have the
		 * 		same length
		 */
		void addSynapses(
				unsigned source,
				const std::vector<unsigned>& targets,
				const std::vector<unsigned>& delays,
				const std::vector<float>& weights,
				const std::vector<unsigned char>& plastic);

		nidx_t maxSourceIdx() const { return m_maxSourceIdx; }
		delay_t maxDelay() const { return m_maxDelay; }
		weight_t maxWeight() const { return m_maxWeight; }
		weight_t minWeight() const { return m_minWeight; }

	private :

		//! \todo perhaps store this as double?
		typedef nemo::Neuron<weight_t> neuron_t;
		std::map<nidx_t, neuron_t> m_neurons;

		typedef Synapse<nidx_t, weight_t> synapse_t;
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
