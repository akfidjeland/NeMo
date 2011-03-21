/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "NeuronType.hpp"

#include <boost/functional/hash.hpp>


namespace nemo {

size_t
hash_value(const nemo::NeuronType& type)
{
	std::size_t seed = 0;
	boost::hash_combine(seed, type.mf_nParam);
	boost::hash_combine(seed, type.mf_nState);
	boost::hash_combine(seed, type.m_name);
	return seed;
}


size_t
NeuronType::hash_value() const
{
	static size_t h = nemo::hash_value(*this);
	return h;	
}

}
