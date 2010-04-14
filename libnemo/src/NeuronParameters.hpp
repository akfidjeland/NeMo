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

class NeuronParameters
{
	public:

		NeuronParameters(const nemo::Network& net, size_t partitionSize);

		void setSigma(class ThalamicInput& th) const;

		float* deviceData() { return md_arr.get(); }

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_allocated; }

		size_t wordPitch() const { return m_wpitch; }

		size_t partitionCount() const;

	private:

		void addNeuron(nidx_t, const nemo::Neuron<float>&);

		void moveToDevice();

		size_t m_partitionSize; // max partition size

		typedef nemo::Neuron<float> neuron_t;
		typedef std::map<nidx_t, neuron_t> acc_t;
		acc_t m_acc;

		boost::shared_ptr<float> md_arr;  // device data

		nidx_t maxNeuronIdx() const;

		size_t m_allocated;

		size_t m_wpitch;

		// max index in each partition
		std::map<pidx_t, nidx_t> m_maxPartitionNeuron;

		void configurePartitionSizes();
};

	} // end namespace cuda
} // end namespace nemo

#endif
