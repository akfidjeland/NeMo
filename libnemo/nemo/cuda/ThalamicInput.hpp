#ifndef THALAMIC_INPUT_HPP
#define THALAMIC_INPUT_HPP

//! \file ThalamicInput.hpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "NVector.hpp"

/*! Parameters for random gaussian current stimulus 
 *
 * Current stimulus as used by Izhikevich in his 2004 paper. The RNG is due to
 * David Thomas.
 *
 * Thalamic input is not alway desirable. It's only enabled if setSigma has
 * been set for at least one partition.
 *
 * \author Andreas Fidjeland
 */

namespace nemo {

	namespace network {
		class NetworkImpl;
	}

	namespace cuda {

class ThalamicInput 
{
	public :

		//! \todo add seed input
		ThalamicInput(const nemo::network::NetworkImpl& net, const class Mapper&);

		/*! \return pointer to device memory containing the RNG state. If
		 * thalamic input is not used, i.e. setSigma has never been called,
		 * return NULL */
		unsigned* deviceRngState() const;

		/*! \return pointer to device memory containing the sigma for each
		 * neuron. If thalamic input is not used, i.e. setSigma has never been
		 * called, return NULL */
		float* deviceSigma() const;

		/*! \return word pitch of both the RNG state vector and the sigma
		 * vector */
		size_t wordPitch() const;

		size_t d_allocated() const;
	
	private :

		NVector<unsigned> m_rngState;

		NVector<float> m_sigma;

		bool m_inUse;
};

	} // end namespace cuda
} // end namespace nemo

#endif
