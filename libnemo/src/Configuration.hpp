#ifndef NEMO_CONFIGURATION_HPP
#define NEMO_CONFIGURATION_HPP

//! \file Configuration.hpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "STDP.hpp"

namespace nemo {

class Configuration
{
	public:

		Configuration();

		/*! Switch on logging and send output to stdout */
		void enableLogging() { m_logging = true; }

		void disableLogging() { m_logging = false; }
		bool loggingEnabled() const { return m_logging; }

		void setCudaMaxPartitionSize(unsigned ps) { m_cudaMaxPartitionSize = ps; }
		unsigned cudaMaxPartitionSize() const { return m_cudaMaxPartitionSize; }

		/*! Set the size of the firing buffer such that it can contain a fixed
		 * number of \a cycles worth of firing data before overflowing. */
		void setCudaFiringBufferLength(unsigned cycles) { m_cudaFiringBufferLength = cycles; }
		unsigned cudaFiringBufferLength() const { return m_cudaFiringBufferLength; }

		/*! Set the cuda device to \a dev. The CUDA library allows the device
		 * to be set only once per thread, so this function may fail if called
		 * multiple times.
		 *
		 * \return
		 * 		-1 if not suitable device found;
		 * 		number of device that will be used, otherwise
		 */
		int setCudaDevice(int dev);

		void setStdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight);

		const STDP<float>* stdpFunction() const { return m_stdpFn; }

	private:

		bool m_logging;
		class STDP<float>* m_stdpFn;

		/* CUDA-specific */
		unsigned m_cudaMaxPartitionSize;
		unsigned m_cudaFiringBufferLength; // in cycles

};


}


#endif
