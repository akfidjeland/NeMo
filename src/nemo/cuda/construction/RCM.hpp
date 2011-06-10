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

	class ConfigurationImpl;

	namespace network {
		class Generator;
	}

	namespace cuda {

		namespace runtime {
			class RCM;
		}

		namespace construction {

class RCM
{
	public :

		/*! Initialise an empty reverse connectivity matrix */
		RCM(const nemo::ConfigurationImpl& conf, const nemo::network::Generator&);

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

		size_t synapseCount() const { return m_synapseCount; }

		/*! Number of words allocated in any enabled RCM fields
		 *
		 * The class maintains the invariant that all RCM fields are either of
		 * this size (if enabled) or WARP_SIZE (if disbled). Furthermore, the
		 * size is always a multiple of WARP_SIZE.
		 */
		size_t size() const;

	private :

		typedef boost::tuple<pidx_t, nidx_t> key;

		typedef boost::unordered_map<key, std::vector<size_t> > warp_map;

		size_t m_synapseCount;

		warp_map m_warps;

		/*! In order to keep track of when we need to start a new warp, store
		 * the number of synapses in each row */
		boost::unordered_map<key, unsigned> m_dataRowLength;

		size_t m_nextFreeWarp;

		/*! Main reverse synapse data: source partition, source neuron, delay */
		std::vector<uint32_t> m_data;
		bool m_useData;

		/*! Forward addressing, word offset into the FCM for each synapse */
		std::vector<uint32_t> m_forward;
		bool m_useForward;

		/*! The weights are \em optionally stored in the reverse format as
		 * well. This is normally not done as the weights are normally used
		 * only in spike delivery which uses the forward form. However, special
		 * neuron type plugins may require this. */
		std::vector<float> m_weights;
		bool m_useWeights;

		/*! Is the RCM in use at all? */
		bool m_enabled;

		/*! If the RCM is in use, do we only keep plastic synapses? See notes
		 * in constructor. */
		bool m_stdpEnabled;


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
