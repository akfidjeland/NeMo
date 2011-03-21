#ifndef NEMO_NETWORK_IMPL_HPP
#define NEMO_NETWORK_IMPL_HPP

//! \file NetworkImpl.hpp

#include <map>
#include <vector>
#include <deque>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>
#include "Axon.hpp"
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

		/*! \copydoc nemo::Network::addNeuron
		 *
		 * \pre the shapes of mf_param and mf_state are identical
		 * \post the shapes of mf_param and mf_state are identical
		 */
		void addNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		/*! \copydoc nemo::Network::setNeuron
		 *
		 * \pre the shapes of mf_param and mf_state are identical
		 * \post the shapes of mf_param and mf_state are identical
		 */
		void setNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

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

		/* Neurons are stored in several Structure-of-arrays, supporting
		 * arbitrary neuron types. Functions modifying these maintain the
		 * invariant that the shapes are the same. */
		std::vector< std::deque<float> > mf_param;
		std::vector< std::deque<float> > mf_state;

		/*! Data are inserted into mf_param etc as they arrive. The mapper
		 * maintains the mapping between global neuron indices, and locations
		 * in the accumulating SoA */
		typedef std::map<nidx_t, size_t> mapper_t;
		mapper_t m_mapper;

		void registerNeuronType(const NeuronType& type);
		std::vector<NeuronType> m_neuronTypes;


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

		size_t existingNeuronLocation(unsigned nidx) const;

		std::deque<float>& f_parameter(size_t i);
		const std::deque<float>& f_parameter(size_t i) const;

		std::deque<float>& f_state(size_t i);
		const std::deque<float>& f_state(size_t i) const;
};

	} // end namespace network
} // end namespace nemo
#endif
