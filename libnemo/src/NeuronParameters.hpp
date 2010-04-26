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

#include "types.hpp"
#include "cuda_types.h"

namespace nemo {

	class Network;

	namespace cuda {

	class Mapper;

class NeuronParameters
{
	public:

		NeuronParameters(const nemo::Network& net,
				const Mapper&,
				size_t partitionSize);

		float* deviceData() { return md_arr.get(); }

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_allocated; }

		size_t wordPitch() const { return m_wpitch; }

		size_t partitionCount() const { return m_partitionCount; }

	private:

		boost::shared_ptr<float> md_arr;  // device data

		size_t m_partitionCount;

		size_t m_allocated;

		size_t m_wpitch;

		typedef nemo::Neuron<float> neuron_t;
		typedef std::map<nidx_t, neuron_t> acc_t;

		void addNeurons(const nemo::Network&, const Mapper&, acc_t*,
				std::map<pidx_t, nidx_t>* maxPartitionNeuron);

		void moveToDevice(const acc_t&, const Mapper&, size_t partitionSize);

		void configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron);
};

	} // end namespace cuda
} // end namespace nemo

#endif
