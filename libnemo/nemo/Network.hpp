#ifndef NEMO_NETWORK_HPP
#define NEMO_NETWORK_HPP

//! \file Network.hpp

#include <vector>
#include <nemo/config.h>
#include <nemo/types.h>

namespace nemo {

namespace mpi {
	class Master;
}

namespace network {
	class NetworkImpl;

}

class Simulation;
class SimulationBackend;
class Configuration;

/*! Networks are constructed by adding individual neurons, and single or groups
 * of synapses to the network. Neurons are given indices (from 0) which should
 * be unique for each neuron. When adding synapses the source or target neurons
 * need not necessarily exist yet, but should be defined before the network is
 * finalised. */
class NEMO_BASE_DLL_PUBLIC Network
{
	public :

		Network();

		~Network();

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
		void addNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		/*! Change parameters/state variables of a single existing neuron
		 *
		 * The parameters are the same as for \a nemo::Network::addNeuron
		 */
		void setNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		/* Add a single synapse and return its unique id */
		synapse_id addSynapse(
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight,
				unsigned char plastic);

		/*!
		 * \param neuron neuron index
		 * \param var variable index
		 * \return state variable \a n.
		 *
		 * For the Izhikevich model the variable indices are 0 = u, 1 = v.
		 */
		float getNeuronState(unsigned neuron, unsigned var) const;

		/*!
		 * \param neuron neuron index
		 * \param param parameter index
		 * \return parameter \a n.
		 *
		 * For the Izhikevich model the parameter indices are 0 = a, 1 = b, 2 = c, 3 = d.
		 */
		float getNeuronParameter(unsigned neuron, unsigned parameter) const;

		/*! Change a single parameter for an existing neuron
		 *
		 * \param neuron neuron index
		 * \param param parameter index
		 * \param val new value of the state variable
		 *
		 * For the Izhikevich model 0 = a, 1 = b, 2 = c, 3 = d
		 */
		void setNeuronParameter(unsigned neuron, unsigned var, float val);

		/*! Change a single state variable for an existing neuron
		 *
		 * \param neuron neuron index
		 * \param var state variable index
		 * \param val new value of the state variable
		 *
		 * For the Izhikevich model variable indices 0 = u, 1 = v
		 */
		void setNeuronState(unsigned neuron, unsigned var, float val);

		/*! \return source neuron id for a given synapse */
		unsigned getSynapseSource(synapse_id id) const;

		/* synapse getter intended for testing only... */

		/*! \return
		 * 		target neurons for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		const std::vector<unsigned>& getTargets(unsigned source) const;

		/*! \return
		 * 		conductance delays for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		const std::vector<unsigned>& getDelays(unsigned source) const;

		/*! \return
		 * 		synaptic weights for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		const std::vector<float>& getWeights(unsigned source) const;

		/*! \return
		 * 		plasticity status for the specified synapses. The reference is
		 * 		valid until the next call to this function.
		 */
		const std::vector<unsigned char>& getPlastic(unsigned source) const;

		unsigned maxDelay() const;

		float maxWeight() const;
		float minWeight() const;

		unsigned neuronCount() const;

	private :

		friend SimulationBackend* simulationBackend(const Network&, const Configuration&);
		friend class nemo::mpi::Master;

		class network::NetworkImpl* m_impl;

		// undefined
		Network(const Network&);
		Network& operator=(const Network&);
};

} // end namespace nemo

#endif
