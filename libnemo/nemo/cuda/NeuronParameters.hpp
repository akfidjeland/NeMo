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
#include <boost/shared_ptr.hpp>

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
		float* d_parameters() const;

		/*! \return device pointer to neuron state data */
		float* d_state() const;

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_allocated; }

		size_t wordPitch() const { return m_wpitch; }

	private:

		boost::shared_ptr<float> md_arr;  // device data

		size_t m_allocated;

		size_t m_wpitch;

		size_t m_pcount;

		size_t allocateDeviceData(size_t pcount, size_t psize);

		void configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron);
};

	} // end namespace cuda
} // end namespace nemo

#endif
