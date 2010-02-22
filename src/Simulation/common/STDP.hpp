#ifndef STDP_HPP
#define STDP_HPP

//! \file STDP.hpp

#include <stdint.h>
#include <vector>

namespace nemo {

/*! \brief User-configurable STDP function */
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

		/*! \return value of STDP function at the given (negative) value of dt */
		T lookupPre(int dt) const;

		/*! \return value of STDP function at the given (positive) value of dt */
		T lookupPost(int dt) const;

		/*! \return length of prefire part of STDP window */
		unsigned int preFireWindow() const { return m_preFireWindow; }

		/*! \return length of postfire part of STDP window */
		unsigned int postFireWindow() const { return m_postFireWindow; }

		/*! \return bit mask indicating which cycles correspond to
		 * potentiation.  LSB = end of STDP window */
		uint64_t potentiationBits() const { return m_potentiationBits; }

		/*! \return bit mask indicating which cycles correspond to depression.  */
		uint64_t depressionBits() const { return m_depressionBits; }

		/*! \return bit mask indicating which cycles correspond to postfire
		 * part of STDP window. */
		uint64_t preFireBits() const { return m_preFireBits; }

		/*! \return bit mask indicating which cycles correspond to prefire
		 * part of STDP window. */
		uint64_t postFireBits() const { return m_postFireBits; }

		/*! \return dt of first spike closest to post-firing, /before/ post-firing. */
		uint closestPreFire(uint64_t arrivals) const;

		/*! \return dt of first spike closest to post-firing, /after/ post-firing. */
		uint closestPostFire(uint64_t arrivals) const;

		/*! \return the STDP function lookup-table */
		const std::vector<T>& function() const { return m_function; }

		bool enabled() const { return m_function.size() > 0; }

		static const unsigned STDP_NO_APPLICATION = unsigned(~0);

	private:

		//! \todo compute the full function only on demand?
		std::vector<T> m_function;

		/* pre-fire part of STDP function, from dt=-1 and down */
		std::vector<T> m_fnPre;

		/* pre-fire part of STDP function, from dt=+1 and up */
		std::vector<T> m_fnPost;

		unsigned int m_preFireWindow;
		unsigned int m_postFireWindow;

		uint64_t m_potentiationBits;
		uint64_t m_depressionBits; 

		uint64_t m_preFireBits;
		uint64_t m_postFireBits;

		T m_maxWeight;
		T m_minWeight;
};

} // namespace nemo

#include "STDP.ipp"

#endif
