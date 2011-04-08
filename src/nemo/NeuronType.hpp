#ifndef NEMO_NEURON_TYPE_HPP
#define NEMO_NEURON_TYPE_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string>
#include <nemo/config.h>

namespace nemo {

/*! \brief General neuron type
 *
 * A neuron type is specified by:
 *
 * - the data it contains
 * - its dynamics 
 *
 * This class is concerned only with the type of data it contains. The
 * simulation data can be set up based on this, regardless of the neuron
 * dynamics.
 */
class NEMO_BASE_DLL_PUBLIC NeuronType
{
	public :

		NeuronType() :
			mf_nParam(0), mf_nState(0),
			m_name("null"), m_membranePotential(0), m_nrand(false) { }

		/*! Create a new neuron model specification
		 * 
		 * This is a generic neuron type which supports a number of run-time
		 * constant parameters and a number of state variables.
		 *
		 * \param f_nParam number of floating point parameters
		 * \param f_nState number of floating point state variables
		 * \param name unique name for this type
		 * \param mp index of membrane potential state variable
		 * \param nrand is a per-neuron gaussian RNG required?
		 */
		NeuronType(size_t f_nParam,
				size_t f_nState,
				const std::string& name,
				unsigned mp,
				bool nrand) :
			mf_nParam(f_nParam), mf_nState(f_nState),
			m_name(name), m_membranePotential(mp), m_nrand(nrand) { }

		size_t f_nParam() const { return mf_nParam; }
		size_t f_nState() const { return mf_nState; }

		size_t hash_value() const;

		/*! Return the name of the neuron model
		 *
		 * This should match the name of the plugin which implements this
		 * neuron type */
		std::string name() const { return m_name; }

		/*! Return the index of the state variable representing the membrane potential */
		unsigned membranePotential() const { return m_membranePotential; }

		bool usesNormalRNG() const { return m_nrand; }

	private :

		size_t mf_nParam;
		size_t mf_nState;
		std::string m_name;

		unsigned m_membranePotential;

		/*! Does this neuron type require a per-neuron gaussian random number generator? */
		bool m_nrand;

		friend size_t hash_value(const nemo::NeuronType&);
};

}

#endif
