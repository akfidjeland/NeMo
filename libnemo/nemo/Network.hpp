#ifndef NEMO_NETWORK_HPP
#define NEMO_NETWORK_HPP

//! \file Network.hpp

#include <vector>
#include <nemo/config.h>

namespace nemo {

namespace mpi {
	class Master;
}

class Simulation;
class SimulationBackend;
class Configuration;
class HardwareConfiguration;

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

		/* Add a single synapse */
		void addSynapse(
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight,
				unsigned char plastic);

		//! \todo change from uchar to bool
		/*! Add a group of synapses to the network
		 *
		 * \param sources
		 * 		Indices of source neuron
		 * \param targets
		 * 		Indices of target neurons
		 * \param delays
		 * 		conductance delays in milliseconds
		 * \param weights
		 * 		synapse weights
		 * \param plastic
		 * 		Specifies for each synapse whether or not it is plastic.
		 * 		See section on STDP.
		 *
		 * \pre
		 * 		\sources, \a targets, \a delays, \a weights, and \a plastic have the
		 * 		same length
		 */
		void addSynapses(
				const std::vector<unsigned>& sources,
				const std::vector<unsigned>& targets,
				const std::vector<unsigned>& delays,
				const std::vector<float>& weights,
				const std::vector<unsigned char>& plastic);

		/* lower-level interface using raw C arrays. This is mainly intended
		 * for use in foreign language interfaces such as C and Mex, where
		 * constructing std::vectors would be redundant. */
		void addSynapses(
				const unsigned source[],
				const unsigned targets[],
				const unsigned delays[],
				const float weights[],
				const unsigned char plastic[],
				size_t len);

		void getSynapses(
				unsigned source,
				std::vector<unsigned>& targets,
				std::vector<unsigned>& delays,
				std::vector<float>& weights,
				std::vector<unsigned char>& plastic) const;

		unsigned maxDelay() const;

		float maxWeight() const;
		float minWeight() const;

		unsigned neuronCount() const;

	private :

		friend NEMO_DLL_PUBLIC Simulation* simulation(const Network& net, const Configuration& conf);
		friend class nemo::mpi::Master;

		class NetworkImpl* m_impl;

		// undefined
		Network(const Network&);
		Network& operator=(const Network&);
};

} // end namespace nemo

#endif
