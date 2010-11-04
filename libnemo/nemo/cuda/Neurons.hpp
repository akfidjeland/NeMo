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

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const;

		/*! \return the word pitch of each per-partition row of data (for a
		 * single parameter/state variable) */
		size_t wordPitch() const;

		/*!
		 * \param parameter PARAM_A, PARAM_B, PARAM_C, or PARAM_D
		 * \return a parameter for a single neuron
		 */
		float getParameter(const DeviceIdx& idx, int parameter) const;

		/* Set parameter for a single neuron
		 *
		 * \param parameter PARAM_A, PARAM_B, PARAM_C, or PARAM_D
		 */
		void setParameter(const DeviceIdx& idx, int parameter, float value);

		/*! Get a state variable for a single neuron
		 *
		 * \param var STATE_U or STATE_V
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device.
		 */
		float getState(const DeviceIdx& idx, int parameter) const;

		/*! Set a state variable for a single neuron
		 *
		 * \param var STATE_U or STATE_V
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device. Additionaly, the next simulation step will involve copying
		 * data from host *to* device.
		 */
		void setState(const DeviceIdx& idx, int var, float value);

		bool rngEnabled() const { return m_rngEnabled; }

	private:

		/* Neuron parameters do not change at run-time (unless the user
		 * specifically does it through \a setParameter) */
		NVector<float, NEURON_FLOAT_PARAM_COUNT> mf_param;

		/* Neuron state variables are updated during simulation. */
		mutable NVector<float, NEURON_FLOAT_STATE_COUNT> mf_state;

		NVector<unsigned, NEURON_UNSIGNED_STATE_COUNT> mu_state;

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