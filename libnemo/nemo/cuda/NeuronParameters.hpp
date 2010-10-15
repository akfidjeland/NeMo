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

		/*! \return device pointer to neuron parameter data */
		float* d_parameters() const { return m_param.deviceData(); }

		/*! \return device pointer to neuron state data */
		float* d_state() const { return m_state.deviceData(); }

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_state.d_allocated() + m_param.d_allocated(); }

		/*! \return the word pitch of each per-partition row of data (for a
		 * single parameter/state variable) */
		size_t wordPitch() const;

	private:

		NVector<float, NEURON_PARAM_COUNT> m_param;
		NVector<float, NEURON_STATE_COUNT> m_state;

		/*! Load vector of the size of each partition onto the device */
		void configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron);
};

	} // end namespace cuda
} // end namespace nemo

#endif
