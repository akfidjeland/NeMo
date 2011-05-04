#ifndef NEMO_CUDA_RUNTIME_RCM_INDEX_HPP
#define NEMO_CUDA_RUNTIME_RCM_INDEX_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/shared_array.hpp>
#include <nemo/cuda/rcm.cu_h>

namespace nemo {
	namespace cuda {

		namespace construction {
			class RcmIndex;
		}

		namespace runtime {


/*! \brief Runtime index into the reverse connectivity matrix  
 *
 * The index is logically a map from neuron to a list of warp numbers (row),
 * where the warp number is an offset into the reverse connectivity matrix.
 *
 * The length of the different rows may differ greatly. In order to save memory
 * the index itself is stored in a compact form where
 *
 * - each row is stored in a contigous chunk of memory
 * - the extent of each row in the index (start and length) is stored in a
 *   separate fixed-size table
 *
 * \see construction::RcmIndex
 */
class RcmIndex
{
	public :

		RcmIndex() : m_allocated(0) {}

		/*! Create an RCM on the device */
		RcmIndex(size_t partitionCount, const construction::RcmIndex&);

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_allocated; }

		/*! \return device pointer to RCM index data */
		const rcm_address_t* d_index() const { return md_data.get(); }

		/*! \return device pointer to the RCM meta index, i.e. the index into the index */
		const rcm_index_address_t* d_metaIndex() const { return md_address.get(); }

	private :

		boost::shared_array<rcm_address_t> md_data;
		boost::shared_array<rcm_index_address_t> md_address;

		/*! Bytes of allocated device memory */
		size_t m_allocated;
};

		}
	}
}

#endif
