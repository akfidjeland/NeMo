#ifndef SYNAPSE_GROUP_HPP
#define SYNAPSE_GROUP_HPP

//! \file SynapseGroup.hpp

#include <stdint.h>

#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <map>

#include "nemo_cuda_types.h"


namespace nemo {

/*! \brief A somewhat arbitrary collection of synapses 
 *
 * On the device synapses are grouped in 2D blocks of memory for all synapses
 * belonging to a particular partition with a particular delay.
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
		 * \param fractionalBits
		 * 		is the number of fractional bits to use in the fixed-point
		 * 		representation of the weights.
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
		size_t fillFcm(
				uint fractionalBits,
				size_t start,
				size_t planeSize,
				std::vector<synapse_t>& h_data);

		/*! \return
		 * 		offset (in terms of number of words) of the specified warp for
		 * 		the given neuron within this group, from the beginning of the
		 * 		FCM */
		uint32_t warpOffset(nidx_t neuron, size_t warp) const;

		weight_t maxAbsWeight() const { return m_maxAbsWeight; }

	private:

		typedef boost::tuple<nidx_t, weight_t> h_synapse_t;
		typedef std::vector<h_synapse_t> row_t;

		/* For each presynaptic neuron we store a row containing all its
		 * outgoing synapses */
		std::map<nidx_t, row_t> mh_synapses;

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

		/* In order to determine the fixed point, we need to keep track of the
		 * range of synapse weights. */
		weight_t m_maxAbsWeight;
};

} // end namespace nemo

#endif
