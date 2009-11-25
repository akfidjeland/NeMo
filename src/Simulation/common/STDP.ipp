//! \file STDP.ipp

#include <algorithm>

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
	m_preFireWindow = prefire.size();
	m_postFireWindow = postfire.size();

	//! \todo check that length is < 64

	// create combined function
	m_function.clear();
	std::copy(prefire.rbegin(), prefire.rend(), std::back_inserter(m_function));
	std::copy(postfire.begin(), postfire.end(), std::back_inserter(m_function));


	//! \todo to test make sure this matches the existing mask
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
