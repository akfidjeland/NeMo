#ifndef STDP_FUNCTION_HPP
#define STDP_FUNCTION_HPP

//! \file StdpFunction.hpp

#include <stdint.h>
#include <vector>
#include <algorithm>

namespace nemo {

/*! \brief User-configurable STDP function */
class StdpFunction
{
	public:

		StdpFunction(unsigned int preFireWindow,
				unsigned int postFireWindow,
				uint64_t potentiationBits,
				uint64_t depressionBits,
				float* stdpFn,
				float maxWeight,
				float minWeight);

		void configureDevice();

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


	private:

		std::vector<float> m_function;

		unsigned int m_preFireWindow;
		unsigned int m_postFireWindow;

		uint64_t m_potentiationBits;
		uint64_t m_depressionBits; 

		float m_maxWeight;
		float m_minWeight;
};



inline
StdpFunction::StdpFunction(unsigned int preFireWindow,
		unsigned int postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		float* stdpFn,
		float maxWeight,
		float minWeight):
	m_function(postFireWindow + preFireWindow, 0.0f),
	m_preFireWindow(preFireWindow),	
	m_postFireWindow(postFireWindow),
	m_potentiationBits(potentiationBits),
	m_depressionBits(depressionBits),
	m_maxWeight(maxWeight),
	m_minWeight(minWeight)
{
	std::copy(stdpFn, stdpFn + m_function.size(), m_function.begin());
}


} // namespace nemo

#endif
