#ifndef NEMO_NETWORK_IMPL_HPP
#define NEMO_NETWORK_IMPL_HPP

//! \file NetworkImpl.hpp

#include <map>
#include <vector>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>
#include "Axon.hpp"
#include "Neurons.hpp"
#include "ReadableNetwork.hpp"

namespace nemo {

	namespace cuda {
		// needed for 'friend' declarations
		class ConnectivityMatrix;
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



class NEMO_BASE_DLL_PUBLIC NetworkImpl : public Generator, public ReadableNetwork
{
	public :

		NetworkImpl();

		/*! Register a new neuron type with the network.
		 *
		 * \return index of the the neuron type, to be used when adding neurons.
		 *
		 * This must be done before neurons of this type can be added to the network.
		 */
		unsigned addNeuronType(const NeuronType&);

		/*! Add a neuron to the network
		 *
		 * \param type index of the neuron type, as returned by \a addNeuronType
		 * \param param floating point parameters of the neuron
		 * \param state floating point state variables of the neuron
		 *
		 * \pre The parameter and state arrays must have the dimensions
		 * 		matching the neuron type represented by \a type.
		 */
		void addNeuron(unsigned type, unsigned idx,
				const float param[], const float state[]);

		/*! Set an existing neuron
		 *
		 * \param param floating point parameters of the neuron
		 * \param state floating point state variables of the neuron
		 *
		 * \pre The parameter and state arrays must have the dimensions
		 * 		matching the neuron type specified when the neuron was first added.
		 */
		void setNeuron(unsigned idx, const float param[], const float state[]);

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

		/*! \copydoc nemo::Network::getSynapseTarget */
		unsigned getSynapseTarget(const synapse_id&) const;

		/*! \copydoc nemo::Network::getSynapseDelay */
		unsigned getSynapseDelay(const synapse_id&) const;

		/*! \copydoc nemo::Network::getSynapseWeight */
		float getSynapseWeight(const synapse_id&) const;

		/*! \copydoc nemo::Network::getSynapsePlastic */
		unsigned char getSynapsePlastic(const synapse_id&) const;

		/*! \copydoc nemo::Network::getSynapsesFrom */
		const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron);

		/* pre: network is not empty */
		nidx_t minNeuronIndex() const;

		/* pre: network is not empty */
		nidx_t maxNeuronIndex() const;

		delay_t maxDelay() const { return m_maxDelay; }
		float maxWeight() const { return m_maxWeight; }
		float minWeight() const { return m_minWeight; }

		unsigned neuronCount() const;

		neuron_iterator neuron_begin() const;
		neuron_iterator neuron_end() const;

		synapse_iterator synapse_begin() const;
		synapse_iterator synapse_end() const;

	private :

		/* Neurons are grouped by neuron type */
		std::vector<Neurons> m_neurons;

		const Neurons& neuronCollection(unsigned type_id) const;
		Neurons& neuronCollection(unsigned type_id);

		/* could use a separate type here, but kept it simple while we use this
		 * type in the neuron_iterator class */
		typedef std::pair<unsigned, unsigned> NeuronAddress;

		typedef std::map<nidx_t, NeuronAddress> mapper_t;
		mapper_t m_mapper;

		/*! Return neuron address of an existing neuron or throw if it does not
		 * exist */
		const NeuronAddress& existingNeuronAddress(unsigned nidx) const;

		/*! \todo consider using unordered here instead, esp. after removing
		 * iterator interface. Currently we need rbegin, which is not found in
		 * unordered_map */
		typedef std::map<nidx_t, Axon> fcm_t;

		fcm_t m_fcm;

		int m_minIdx;
		int m_maxIdx;
		delay_t m_maxDelay;
		float m_minWeight;
		float m_maxWeight;

		//! \todo modify public interface to avoid friendship here
		friend class nemo::cuda::ConnectivityMatrix;
		friend class nemo::cuda::ThalamicInput;
		friend class nemo::ConnectivityMatrix;
		friend class nemo::cpu::Simulation;
		friend class nemo::mpi::Master;

		friend class programmatic::synapse_iterator;

		/*! Internal buffer for synapse queries */
		std::vector<synapse_id> m_queriedSynapseIds;

		const Axon& axon(nidx_t source) const;

};

	} // end namespace network
} // end namespace nemo

#endif
