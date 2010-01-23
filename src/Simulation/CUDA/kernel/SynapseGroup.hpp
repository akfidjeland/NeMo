#ifndef SYNAPSE_GROUP_HPP
#define SYNAPSE_GROUP_HPP

//! \file SynapseGroup.hpp

#include <stdint.h>

#include <boost/shared_ptr.hpp>
#include <vector>
#include <map>

#include "nemo_cuda_types.h"


/*! \brief A somewhat arbitrary collection of synapses 
 *
 * On the device synapses are grouped in 2D blocks of memory for all synapses
 * belonging to a particular partition with a particular delay.
 *
 */

class SynapseGroup 
{
	public:

		/*! Create an empty synapse group */
		SynapseGroup();

		/*! Add a single synapse to the synapse group
		 *
		 * \return
		 * 		Index (location within a row) of the synapse that was just adde
		 */
		sidx_t addSynapse(nidx_t sourceNeuron,
				pidx_t targetPartition,
				nidx_t targetNeuron,
				float weight,
				uchar plastic);

		/*! Add several synapses with the same source neuron */
		void addSynapses(nidx_t sourceNeuron,
				size_t ncount,
				const pidx_t targetPartition[],
				const nidx_t targetNeuron[],
				const float weight[],
				const uchar plastic[]);

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

		/*! Get weights for a particular neuron in the form of 3 vectors
		 * (partition, neuron, weight).
		 * \return
		 * 		length of vectors
		 */
		size_t getWeights(nidx_t sourceNeuron,
				uint currentCycle,
				pidx_t* partition[],
				nidx_t* neuron[],
				weight_t* weight[],
				uchar* plastic[]);

		/*! fill host buffer with synapse data.
		 *
		 * \param start
		 * 		word offset withing host data, for the first unused warp
		 * \param planeSize
		 * 		size of a plane (in words) of the FCM, i.e. the size of all the
		 * 		address /or/ weight data. This is the distance between a synapse's
		 * 		address data and its weight data.
		 *
		 * \return
		 * 		number of consecutive words written (starting at \a start)
		 */
		size_t fillFcm(size_t start,
				size_t planeSize,
				std::vector<synapse_t>& h_data);

		/*! \return
		 * 		offset (in terms of number of words) of the specified warp for
		 * 		the given neuron within this group, from the beginning of the
		 * 		FCM */
		size_t warpOffset(nidx_t neuron, size_t warp) const;

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

		/* The user may want to read back the modified weight matrix. We then
		 * need the corresponding non-compressed addresses as well. The shape
		 * of each of these is exactly that of the weights on the device. */
		std::map<int, std::vector<pidx_t> > mf_targetPartition;
		std::map<int, std::vector<nidx_t> > mf_targetNeuron;
		std::map<int, std::vector<uchar> > mf_plastic;
		std::vector<synapse_t> mf_weights; // in device format

		/* We make sure to only copy each datum at most once per cycle */
		uint m_lastSync;

		std::map<nidx_t, size_t> m_warpOffset;  // within FCM of the /first/ warp for any neuron
};

#endif
