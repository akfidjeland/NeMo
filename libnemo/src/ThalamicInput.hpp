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
	namespace cuda {

class ThalamicInput 
{
	public :

		ThalamicInput(size_t partitionCount, size_t partitionSize, int seed);

		/*! Set values of sigma for a single neuron */
		void setNeuronSigma(size_t partition, size_t neuron, float val);

		/*! Move data to device and clear host-side data */
		void moveToDevice();

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

		size_t m_partitionCount;

		size_t m_partitionSize;

		int m_seed;

		void initRngState();
};

	} // end namespace cuda
} // end namespace nemo

#endif
