#ifndef THALAMIC_INPUT_HPP
#define THALAMIC_INPUT_HPP

//! \file ThalamicInput.hpp

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

class ThalamicInput 
{
	public :

		ThalamicInput(size_t partitionCount, size_t partitionSize, int seed);

		/*! Set values of sigma for a single partition */
		void setSigma(size_t partition, const float* arr, size_t length);

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

#endif
