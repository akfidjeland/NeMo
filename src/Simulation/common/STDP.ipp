//! \file STDP.ipp

#include <algorithm>
#include <stdexcept>
#include <assert.h>

#include "bitops.h"

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



template<typename T>
uint
STDP<T>::closestPreFire(uint64_t arrivals) const
{
	uint64_t validArrivals = arrivals & m_preFireBits;
	int dt =  ctz64(validArrivals >> m_postFireWindow);
	return validArrivals ? (uint) dt : STDP_NO_APPLICATION;
}



template<typename T>
uint
STDP<T>::closestPostFire(uint64_t arrivals) const
{
	uint64_t validArrivals = arrivals & m_postFireBits;
	int dt = clz64(validArrivals << uint64_t(64 - m_postFireWindow));
	return validArrivals ? (uint) dt : STDP_NO_APPLICATION;
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
		throw std::runtime_error("size of STDP window too large");
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



/* Helper function for c interfaces */
template<typename T>
void
configure_stdp(
		STDP<T>& stdp,
		size_t pre_len,
		size_t post_len,
		T* pre_fn,
		T* post_fn,
		T maxWeight,
		T minWeight)
{
	std::vector<T> prefire;
	std::copy(pre_fn, pre_fn + pre_len, std::back_inserter(prefire));

	std::vector<T> postfire;
	std::copy(post_fn, post_fn + post_len, std::back_inserter(postfire));

	stdp.configure(prefire, postfire, minWeight, maxWeight);
}

} // end namespace nemo
