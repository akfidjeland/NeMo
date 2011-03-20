#ifndef NEMO_CUDA_NEURONS_HPP
#define NEMO_CUDA_NEURONS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file Neurons.hpp

#include <map>

#include "Mapper.hpp"
#include "NVector.hpp"
#include "Bitvector.hpp"
#include "kernel.cu_h"
#include "types.h"

namespace nemo {

	namespace network {
		class Generator;
	}

	namespace cuda {

	class Mapper;

/*! Per-neuron device data
 *
 * Per-neuron data is split into parameters and state variables. The former are
 * fixed at run-time while the latter is modififed by the simulation.
 *
 * Additionally we split data into floating-point data and (unsigned) integer
 * data, which are indicated with prefixes 'f' and 'u' respectively.
 *
 * In the current implementation only floating point parameters and state
 * variables can be read or written. The need for doing this with integer data
 * does not arise when using Izhikevich neurons.
 */
class Neurons
{
	public:

		Neurons(const network::Generator& net, Mapper&);

		/*! Perform any required synchronisation between host and device data.
		 * Such synchronisation may be required if the user has requested that
		 * the data should be updated. The step function should be called for
		 * every simulation cycle. */
		void step(cycle_t cycle);

		/*! \return device pointer to floating-point neuron parameter data */
		float* df_parameters() const { return mf_param.deviceData(); }

		/*! \return device pointer to floating-point neuron state data */
		float* df_state() const { return mf_state.deviceData(); }

		/*! \return device pointer to unsigned neuron state data */
		unsigned* du_state() const { return mu_state.deviceData(); }

		/*! \return device pointer to bit vector with only valid/existing neurons set */
		uint32_t* d_valid() const { return m_valid.d_data(); }

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const;

		/*! \return the word pitch of each per-partition row of data (for a
		 * single parameter/state variable) */
		size_t wordPitch32() const;

		/*! \return the word pitch of bitvectors */
		size_t wordPitch1() const { return m_valid.wordPitch(); }

		/*!
		 * \param idx neuron index
		 * \param parameter PARAM_A, PARAM_B, PARAM_C, or PARAM_D
		 * \return a parameter for a single neuron
		 */
		float getParameter(const DeviceIdx& idx, unsigned parameter) const;

		/* Set parameter for a single neuron
		 *
		 * \param idx neuron index
		 * \param parameter PARAM_A, PARAM_B, PARAM_C, or PARAM_D
		 * \param value
		 */
		void setParameter(const DeviceIdx& idx, unsigned parameter, float value);

		/*! Get a state variable for a single neuron
		 *
		 * \param idx neuron index
		 * \param var STATE_U or STATE_V
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device.
		 */
		float getState(const DeviceIdx& idx, unsigned var) const;

		/*! Set a state variable for a single neuron
		 *
		 * \param idx neuron index
		 * \param var STATE_U or STATE_V
		 * \param value
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device. Additionaly, the next simulation step will involve copying
		 * data from host *to* device.
		 */
		void setState(const DeviceIdx& idx, unsigned var, float value);

		bool rngEnabled() const { return m_rngEnabled; }

	private:

		const unsigned mf_nParams;
		const unsigned mf_nStateVars;
		const unsigned mu_nStateVars;

		/* Neuron parameters do not change at run-time (unless the user
		 * specifically does it through \a setParameter) */
		NVector<float> mf_param;

		/* Neuron state variables are updated during simulation. */
		mutable NVector<float> mf_state;

		NVector<unsigned> mu_state;

		/* In the translation from global neuron indices to device indices,
		 * there may be 'holes' left in the index space. The valid bitvector
		 * specifies which neurons are valid/existing, so that the kernel can
		 * ignore these. Of course, the ideal situation is that the index space
		 * is contigous, so that warp divergence is avoided on the device */
		Bitvector m_valid;

		cycle_t m_cycle;
		mutable cycle_t mf_lastSync;

		bool mf_paramDirty;
		bool mf_stateDirty;

		/*! Load vector of the size of each partition onto the device */
		void configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron);

		/*! Read the neuron state from the device, if it the device data is not
		 * already cached on the host */
		void readStateFromDevice() const; // conceptually const, this is just caching

		bool m_rngEnabled;
};


	} // end namespace cuda
} // end namespace nemo

#endif
