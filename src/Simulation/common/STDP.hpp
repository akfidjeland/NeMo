#ifndef STDP_HPP
#define STDP_HPP

//! \file STDP.hpp

#include <stdint.h>
#include <vector>

namespace nemo {

/*! \brief User-configurable STDP function */
//! \todo template this for different FT types
template<typename T>
class STDP
{
	public:

		STDP() :
			m_preFireWindow(0),
			m_postFireWindow(0),
			m_potentiationBits(0),
			m_depressionBits(0),
			m_maxWeight(0.0),
			m_minWeight(0.0)
		{}

		STDP(const std::vector<T>& prefire, const std::vector<T>& postfire,
				T minWeight, T maxWeight);

		void configure(
				const std::vector<T>& prefire,
				const std::vector<T>& postfire,
				T minWeight,
				T maxWeight);

		T maxWeight() const { return m_maxWeight; }
		T minWeight() const { return m_minWeight; }

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
		const std::vector<T>& function() const { return m_function; }

		bool enabled() const { return m_function.size() > 0; }

	private:

		std::vector<T> m_function;

		unsigned int m_preFireWindow;
		unsigned int m_postFireWindow;

		uint64_t m_potentiationBits;
		uint64_t m_depressionBits; 

		T m_maxWeight;
		T m_minWeight;
};

} // namespace nemo

#include "STDP.ipp"

#endif
