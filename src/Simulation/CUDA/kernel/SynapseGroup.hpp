#ifndef SYNAPSE_GROUP_HPP
#define SYNAPSE_GROUP_HPP

//! \file SynapseGroup.hpp

#include <stdint.h>

#include <boost/shared_ptr.hpp>
#include <vector>
#include <map>

#include <nemo_types.h>


/*! \brief A somewhat arbitrary collection of synapses 
 *
 * On the device synapses are grouped in 2D blocks of memory for all synapses
 * belonging to a particular partition with a particular delay.
 *
 */

class SynapseGroup 
{
	public:
		
		//! \todo move this into separate file and share across project
		typedef uint pidx_t;
		//typedef uint32_t pidx_t;
		//typedef uint32_t nidx_t;
		typedef float weight_t;

		/*! Create an empty synapse group */
		SynapseGroup();

		/*! Add a single synapse */
		void addSynapse(nidx_t sourceNeuron,
				pidx_t targetPartition,
				nidx_t targetNeuron,
				float weight);

		/*! Add several synapses with the same source neuron */
		void addSynapses(nidx_t sourceNeuron,
				size_t ncount,
				const pidx_t targetPartition[],
				const nidx_t targetNeuron[],
				const float weight[]);

		/*! Move to device and free host data. Return pointer to device data.*/
		boost::shared_ptr<uint32_t> moveToDevice();

		/* There are two planes (one for addresses and one for weights, the
		 * size of which can be determined based on the (fixed) partition size
		 * and the pitch */
		size_t planeSize() const;

		/*! \return total size of data (in bytes) on the device */
		size_t dataSize() const;

		/*! \return row pitch on the device (in words) */
		size_t wpitch() const;

		/*! \return row pitch on the device (in bytes) */
		size_t bpitch() const;

		/* On the device both address and weight data is squeezed into 32b */
		typedef uint32_t synapse_t;

		/*! \return address of synapse group on device */
		synapse_t* d_address() const { return md_synapses.get(); }

		/*! \return number of bytes allocated on device */
		size_t d_allocated() const { return m_allocated; }

	private:

		struct Row {
			std::vector<synapse_t> addresses; 
			std::vector<weight_t> weights;
		};

		/* For each presynaptic neuron we store a row containing all its
		 * outgoing synapses */
		std::map<nidx_t, Row> mh_synapses;

		/*! On the device, the synapses are stored one row per presynaptic
		 * neurons, with a fixed row pitch.  Any padding is at the end of the
		 * row */
		boost::shared_ptr<synapse_t> md_synapses;
		size_t md_bpitch;

		size_t m_allocated; // bytes on device

		size_t maxSynapsesPerNeuron() const;
};

#endif
