#ifndef NEURON_PARAMETERS_HPP
#define NEURON_PARAMETERS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file NeuronParameters.hpp

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

class NeuronParameters
{
	public:

		NeuronParameters(const network::Generator& net, Mapper&);

		/*! Perform any required synchronisation between host and device data.
		 * Such synchronisation may be required if the user has requested that
		 * the data should be updated. The step function should be called for
		 * every simulation cycle. */
		void step(cycle_t cycle);

		/*! \return device pointer to neuron parameter data */
		float* d_parameters() const { return m_param.deviceData(); }

		/*! \return device pointer to neuron state data */
		float* d_state() const { return m_state.deviceData(); }

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_state.d_allocated() + m_param.d_allocated(); }

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

	private:

		/* Neuron parameters do not change at run-time (unless the user
		 * specifically does it through \a setParameter) */
		NVector<float, NEURON_PARAM_COUNT> m_param;

		/* Neuron state variables are updated during simulation. */
		mutable NVector<float, NEURON_STATE_COUNT> m_state;

		cycle_t m_cycle;
		mutable cycle_t m_lastSync;

		bool m_paramDirty;
		bool m_stateDirty;

		/*! Load vector of the size of each partition onto the device */
		void configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron);

		/*! Read the neuron state from the device, if it the device data is not
		 * already cached on the host */
		void readStateFromDevice() const; // conceptually const, this is just caching
};


	} // end namespace cuda
} // end namespace nemo

#endif
