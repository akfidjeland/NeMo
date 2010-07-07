/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file STDP.ipp

#include <algorithm>
#include <stdexcept>
#include <assert.h>

#include "exception.hpp"

namespace nemo {


template<typename T>
STDP<T>::STDP(
			const std::vector<T>& prefire,
			const std::vector<T>& postfire,
			T minWeight,
			T maxWeight)
{
	configure(prefire, postfire, minWeight, maxWeight);
}



template<typename T>
T
STDP<T>::lookupPre(int dt) const
{
	assert(dt >= 0);
	assert(dt < m_preFireWindow);
	return m_fnPre.at(dt);
}



template<typename T>
T
STDP<T>::lookupPost(int dt) const
{
	assert(dt >= 0);
	assert(dt < m_postFireWindow);
	return m_fnPost.at(dt);
}



inline
void
setBit(size_t bit, uint64_t& word)
{
	word = word | (uint64_t(1) << bit);
}


template<typename T>
void
STDP<T>::configure(
			const std::vector<T>& prefire,
			const std::vector<T>& postfire,
			T minWeight,
			T maxWeight)
{
	m_fnPre = prefire;
	m_fnPost = postfire;

	m_preFireWindow = prefire.size();
	m_postFireWindow = postfire.size();

	/*! \todo This constraint is too weak. Also need to consider max delay in
	 * network here */
	if(m_preFireWindow + m_postFireWindow > 64) {
		throw nemo::exception(NEMO_INVALID_INPUT, "size of STDP window too large");
	}

	// create combined function
	m_function.clear();
	std::copy(prefire.rbegin(), prefire.rend(), std::back_inserter(m_function));
	std::copy(postfire.begin(), postfire.end(), std::back_inserter(m_function));

	m_potentiationBits = 0;
	m_depressionBits = 0;

	int bit=0;
	for(typename std::vector<T>::reverse_iterator f = m_function.rbegin();
			f != m_function.rend(); ++bit, ++f) {
		if(*f > 0.0) {
			setBit(bit, m_potentiationBits);
		} else if (*f < 0.0) {
			setBit(bit, m_depressionBits);
		}
	}

	m_minWeight = minWeight;
	m_maxWeight = maxWeight;

	m_preFireBits = (~(uint64_t(~0) << uint64_t(preFireWindow()))) << uint64_t(postFireWindow());
	m_postFireBits = ~(uint64_t(~0) << uint64_t(postFireWindow()));
}


} // end namespace nemo
