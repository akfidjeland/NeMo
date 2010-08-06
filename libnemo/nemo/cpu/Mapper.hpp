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

#include <nemo/types.hpp>
#include <nemo/NetworkImpl.hpp>

namespace nemo {
	namespace cpu {

class Mapper
{
	public :

		Mapper(const nemo::NetworkImpl& net) :
			m_offset(0),
			m_ncount(0)
		{
			if(net.neuronCount() > 0) {
				m_offset = net.minNeuronIndex();
				m_ncount = net.maxNeuronIndex() - net.minNeuronIndex() + 1;
			}
		}

		/* Convert global neuron index to local */
		nidx_t localIdx(nidx_t global) const {
			assert(global >= m_offset);
			assert(global - m_offset < m_ncount);
			return global - m_offset;
		}
		
		/* Convert local neuron index to global */
		nidx_t globalIdx(nidx_t local) const {
			assert(local < m_ncount);
			return local + m_offset;
		}

		/* Return number of valid neuron *indices*. The actual number of
		 * neurons may be smaller as some indices may correspond to inactive
		 * neurons */ 
		unsigned neuronCount() const {
			return m_ncount;
		}

		nidx_t minLocalIdx() const {
			return m_offset;
		}

		nidx_t maxLocalIdx() const {
			return m_offset + m_ncount - 1;
		}

	private :

		nidx_t m_offset;

		unsigned m_ncount;
};

	}
}

#endif
