#ifndef NEMO_CPU_MAPPER_HPP
#define NEMO_CPU_MAPPER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <set>
#include <cassert>

#include <nemo/types.hpp>
#include <nemo/network/Generator.hpp>
#include <nemo/Mapper.hpp>

namespace nemo {
	namespace cpu {

/*! \copydoc nemo::Mapper */
class Mapper : public nemo::Mapper<nidx_t, nidx_t>
{
	public :

		Mapper(const network::Generator& net) :
			m_offset(0),
			m_ncount(0)
		{
			if(net.neuronCount() > 0) {
				m_offset = net.minNeuronIndex();
				m_ncount = net.maxNeuronIndex() - net.minNeuronIndex() + 1;
			}
		}

		/*! Add a neuron to the set of existing neurons and return the local
		 * index. */
		nidx_t addGlobal(const nidx_t& global);

		/*! \return local neuron index corresponding to the given global neuron index
		 *
		 * The global neuron index refers to a neuron which may or may not exist.
		 *
		 * \see existingDeviceIdx for a safer function.
		 */
		nidx_t localIdx(const nidx_t& global) const {
			return global - m_offset;
		}

		/*! \return
		 * 		local index corresponding to the given global index of an
		 * 		existing neuron
		 *
		 * Throw if the neuron does not exist
		 *
		 * \see localIdx
		 */
		nidx_t existingLocalIdx(const nidx_t& global) const;
		
		/* Convert local neuron index to global */
		nidx_t globalIdx(const nidx_t& local) const {
			assert(local < m_ncount);
			return local + m_offset;
		}

		/*! \return
		 * 		number of neurons in the valid index range. This may be larger
		 * 		than the existing number of neurons in this range as some
		 * 		indices may be unused.*/
		unsigned neuronsInValidRange() const {
			return m_ncount;
		}

		nidx_t minLocalIdx() const {
			return m_offset;
		}

		nidx_t maxLocalIdx() const {
			return m_offset + m_ncount - 1;
		}

		bool existingGlobal(const nidx_t& global) const {
			return m_existingGlobal.count(global) == 1;
		}

		bool existingLocal(const nidx_t& local) const {
			return existingGlobal(globalIdx(local));
		}

	private :

		nidx_t m_offset;

		unsigned m_ncount;

		std::set<nidx_t> m_existingGlobal;
};

	}
}

#endif
