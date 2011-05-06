#ifndef NEMO_CUDA_CONSTRUCTION_RCM_HPP
#define NEMO_CUDA_CONSTRUCTION_RCM_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>

#include <nemo/types.hpp>
//! \todo move DeviceIdx to types.hpp
#include <nemo/cuda/Mapper.hpp>
#include <nemo/cuda/types.h>


namespace nemo {
	namespace cuda {

		namespace runtime {
			class RCM;
		}

		namespace construction {

class RCM
{
	public :

		explicit RCM(bool useWeights);

		/*! Add a new synapse to the reverse connectivity matrix
		 *
		 * \param synapse full synapse
		 * \param d_source index of source neuron on device
		 * \param d_target index of target neuron on device
		 * \param f_addr word address of this synapse in the forward matrix
		 */
		void addSynapse(const Synapse& synapse,
				const DeviceIdx& d_source,
				const DeviceIdx& d_target,
				size_t f_addr);

	private :

		typedef boost::tuple<pidx_t, nidx_t> key;

		typedef boost::unordered_map<key, std::vector<size_t> > warp_map;

		warp_map m_warps;

		/*! In order to keep track of when we need to start a new warp, store
		 * the number of synapses in each row */
		boost::unordered_map<key, unsigned> m_dataRowLength;

		size_t m_nextFreeWarp;

		/*! Main reverse synapse data: source partition, source neuron, delay */
		std::vector<uint32_t> m_data;

		/*! Forward addressing, word offset into the FCM for each synapse */
		std::vector<uint32_t> m_forward;

		/*! The weights are \em optionally stored in the reverse format as
		 * well. This is normally not done as the weights are normally used
		 * only in spike delivery which uses the forward form. However, special
		 * neuron type plugins may require this. */
		std::vector<float> m_weights;
		bool m_useWeights;

		/*! Allocate space for a new RCM synapse for the given (target) neuron.
		 *
		 * \return
		 * 		word offset for the synapse. This is the same for all the different
		 * 		planes of data.
		 */
		size_t allocateSynapse(const DeviceIdx& target);

		friend class runtime::RCM;
};



		} // end namespace construction
	} // end namespace cuda
} // end namespace nemo


#endif
