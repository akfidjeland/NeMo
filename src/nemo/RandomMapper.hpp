#ifndef NEMO_COMPACTING_MAPPER_HPP
#define NEMO_COMPACTING_MAPPER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/bimap.hpp>
#include <boost/format.hpp>

#include <nemo/Mapper.hpp>
#include <nemo/exception.hpp>

namespace nemo {

/*! Mapper between global neuron index space and another index space
 *
 * The user of this class is responsible for providing both indices
 */
template<class L>
class RandomMapper
{
	private :

		typedef boost::bimap<nidx_t, L> bm_type;

	public :

		/*! Add a new global/local neuron index pair */
		void insert(nidx_t gidx, const L& lidx) {
			m_bm.insert(typename bm_type::value_type(gidx, lidx));
		}

		/*! \return local index corresponding to the global neuron index \a gidx 
		 *
		 * \throws nemo::exception if the global neuron does not exist
		 */
		const L& localIdx(const nidx_t& gidx) const {
			using boost::format;
			try {
				return m_bm.left.at(gidx);
			} catch(std::out_of_range) {
				throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Non-existing neuron index %u") % gidx));
			} 
		}

		/*! Iterator over <global,local> pairs */
		typedef typename bm_type::left_const_iterator const_iterator;

		const_iterator begin() const { return m_bm.left.begin(); }
		const_iterator end() const { return m_bm.left.end(); }

	private :

		bm_type m_bm;
};


}

#endif
