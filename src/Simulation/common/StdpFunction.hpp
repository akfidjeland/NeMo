#ifndef STDP_FUNCTION_HPP
#define STDP_FUNCTION_HPP

//! \file StdpFunction.hpp

#include <stdint.h>
#include <vector>
#include <algorithm>

namespace nemo {

/*! \brief User-configurable STDP function */
//! \todo template this for different FT types
class StdpFunction
{
	public:

		StdpFunction() :
			m_preFireWindow(0),
			m_postFireWindow(0),
			m_potentiationBits(0),
			m_depressionBits(0),
			m_maxWeight(0.0),
			m_minWeight(0.0)
		{}

		StdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight);

		void configure(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight);

		float maxWeight() const { return m_maxWeight; }

		float minWeight() const { return m_minWeight; }

		/*! \return length of prefire part of STDP window */
		unsigned int preFireWindow() const { return m_preFireWindow; }

		/*! \return length of postfire part of STDP window */
		unsigned int postFireWindow() const { return m_postFireWindow; }

		/*! \return bit mask indicating which cycles correspond to
		 * potentiation.  LSB = end of STDP window */
		uint64_t potentiationBits() const { return m_potentiationBits; }

		/*! \return bit mask indicating which cycles correspond to depression.  */
		uint64_t depressionBits() const { return m_depressionBits; }

		/*! \return the STDP function lookup-table */
		const std::vector<float>& function() const { return m_function; }

		bool enabled() const { return m_function.size() > 0; }

	private:

		//! \todo remove
		std::vector<float> m_function;

		//std::vector<float> m_prefire;
		//std::vector<float> m_postfire;

		unsigned int m_preFireWindow;
		unsigned int m_postFireWindow;

		uint64_t m_potentiationBits;
		uint64_t m_depressionBits; 

		float m_maxWeight;
		float m_minWeight;
};



inline
StdpFunction::StdpFunction(
			const std::vector<float>& prefire,
			const std::vector<float>& postfire,
			float minWeight,
			float maxWeight)
{
	fprintf(stderr, "STDP ctor");
	configure(prefire, postfire, minWeight, maxWeight);
	fprintf(stderr, "STDP ctor end");
}



inline
void
setBit(size_t bit, uint64_t& word)
{
	word = word | (uint64_t(1) << bit);
}


inline
void
StdpFunction::configure(
			const std::vector<float>& prefire,
			const std::vector<float>& postfire,
			float minWeight,
			float maxWeight)
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
	for(std::vector<float>::reverse_iterator f = m_function.rbegin();
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
inline
void
configure_stdp(
		StdpFunction& stdp,
		size_t pre_len,
		size_t post_len,
		float* pre_fn,
		float* post_fn,
		float maxWeight,
		float minWeight)
{
	std::vector<float> prefire;
	std::copy(pre_fn, pre_fn + pre_len, std::back_inserter(prefire));

	std::vector<float> postfire;
	std::copy(post_fn, post_fn + post_len, std::back_inserter(postfire));

	stdp.configure(prefire, postfire, minWeight, maxWeight);
}



} // namespace nemo

#endif
