#ifndef NEMO_STDP_FUNCTION
#define NEMO_STDP_FUNCTION

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <nemo/config.h>
#include <nemo/internal_types.h>

#ifdef NEMO_MPI_ENABLED

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace boost {
	namespace serialization {
		class access;
	}
}

#endif

namespace nemo {


/*! \brief User-configurable STDP function */
class NEMO_BASE_DLL_PUBLIC StdpFunction
{
	public :

		StdpFunction() : m_minWeight(0.0), m_maxWeight(0.0) { }

		StdpFunction(const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight, float maxWeight);

		/* pre-fire part of STDP function, from dt=-1 and down */
		const std::vector<float>& prefire() const { return m_prefire; }

		/* pre-fire part of STDP function, from dt=+1 and up */
		const std::vector<float>& postfire() const { return m_postfire; }

		float minWeight() const { return m_minWeight; }

		float maxWeight() const { return m_maxWeight; }

		/*! \return bit mask indicating which cycles correspond to
		 * potentiation.  LSB = end of STDP window. */
		uint64_t potentiationBits() const;

		/*! \return bit mask indicating which cycles correspond to depression.
		 * LSB = end of STDP window. */
		uint64_t depressionBits() const;

	private :

		std::vector<float> m_prefire;

		std::vector<float> m_postfire;

		float m_minWeight;

		float m_maxWeight;

		uint64_t getBits(bool (*pred)(float)) const;

#ifdef NEMO_MPI_ENABLED
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & m_prefire;
			ar & m_postfire;
			ar & m_maxWeight;
			ar & m_minWeight;
		}
#endif
};

}

#endif
