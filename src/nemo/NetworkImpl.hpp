#ifndef NEMO_NETWORK_IMPL_HPP
#define NEMO_NETWORK_IMPL_HPP

//! \file NetworkImpl.hpp

#include <map>
#include <vector>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>
#include "Axon.hpp"

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

		/*! \copydoc nemo::Network::addNeuron */
		void addNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		void addNeuron(nidx_t nidx, const Neuron<float>&);

		/*! \copydoc nemo::Network::setNeuron */
		void setNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		void setNeuron(nidx_t nidx, const Neuron<float>& n);

		/*! \copydoc nemo::Network::addSynapse */
		synapse_id addSynapse(
				unsigned source,
				unsigned target,
				unsigned delay,
				float weight,
				unsigned char plastic);

		/*! \copydoc nemo::Network::getNeuronState */
		float getNeuronState(unsigned neuron, unsigned var) const;

		/*! \copydoc nemo::Network::getNeuronParameter */
		float getNeuronParameter(unsigned neuron, unsigned parameter) const;

		/*! \copydoc nemo::Network::setNeuronState */
		void setNeuronParameter(unsigned neuron, unsigned var, float val);

		/*! \copydoc nemo::Network::setNeuronParameter */
		void setNeuronState(unsigned neuron, unsigned var, float val);

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

		/*! \return neuron with given index
		 * Throws if neuron does not exist
		 */
		const neuron_t& getNeuron(unsigned idx) const;
		neuron_t& getNeuron(unsigned idx);

		/*! \todo consider using unordered here instead, esp. after removing
		 * iterator interface. Currently we need rbegin, which is not found in
		 * unordered_map */
		typedef std::map<nidx_t, Axon> fcm_t;

		fcm_t m_fcm;

		int m_minIdx;
		int m_maxIdx;
		delay_t m_maxDelay;
		weight_t m_minWeight;
		weight_t m_maxWeight;

		//! \todo modify public interface to avoid friendship here
		friend class nemo::cuda::ConnectivityMatrix;
		friend class nemo::cuda::NeuronParameters;
		friend class nemo::cuda::ThalamicInput;
		friend class nemo::ConnectivityMatrix;
		friend class nemo::cpu::Simulation;
		friend class nemo::mpi::Master;

		friend class programmatic::synapse_iterator;

		/* Internal buffers for synapse queries */
		mutable std::vector<unsigned> m_queriedTargets;
		mutable std::vector<unsigned> m_queriedDelays;
		mutable std::vector<float> m_queriedWeights;
		mutable std::vector<unsigned char> m_queriedPlastic;

		const Axon& axon(nidx_t source) const;
};

	} // end namespace network
} // end namespace nemo
#endif
