#ifndef STDP_HPP
#define STDP_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file STDP.hpp

#include <vector>

#include <nemo_config.h>
#include "types.h"

#ifdef INCLUDE_MPI

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace boost {
	namespace serialization {
		class access;
	}
}

#endif

namespace nemo {

template<typename T> class STDP;
template<typename T> void check_close(const STDP<T>&, const STDP<T>&);


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

		/*! \return the STDP function lookup-table */
		const std::vector<T>& function() const { return m_function; }

		bool enabled() const { return m_function.size() > 0; }

		static const unsigned STDP_NO_APPLICATION = unsigned(~0);

	private:

		 friend void check_close<>(const nemo::STDP<T>&, const nemo::STDP<T>&);

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

#ifdef INCLUDE_MPI
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & m_function;
			ar & m_fnPre;
			ar & m_fnPost;
			ar & m_preFireWindow;
			ar & m_postFireWindow;
			ar & m_potentiationBits;
			ar & m_depressionBits;
			ar & m_preFireBits;
			ar & m_postFireBits;
			ar & m_maxWeight;
			ar & m_minWeight;
		}
#endif
};

} // namespace nemo


#include "STDP.ipp"

#endif
