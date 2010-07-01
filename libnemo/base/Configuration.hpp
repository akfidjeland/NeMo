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

#include "nemo_config.h"
#include <ostream>
#include <vector>


namespace nemo {
	class Configuration;
}

std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf);


namespace nemo {

	class SimulationBackend;

	namespace mpi {
		class Master;
		class Worker;
	}

class DLL_PUBLIC Configuration
{
	public:

		Configuration();

		Configuration(const Configuration&);

		~Configuration();

		/*! Switch on logging and send output to stdout */
		void enableLogging(); 

		void disableLogging();
		bool loggingEnabled() const;

		void setCudaPartitionSize(unsigned ps); 
		unsigned cudaPartitionSize() const;

		/*! Set the size of the firing buffer such that it can contain a fixed
		 * number of \a cycles worth of firing data before overflowing. */
		void setCudaFiringBufferLength(unsigned cycles); 
		unsigned cudaFiringBufferLength() const;

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


		/*! The simulation uses a fixed-point format internally for synaptics
		 * weights. Call this method to specify how many fractional bits to
		 * use. If nothing is specified the backend chooses a sensible value
		 * based on the range of weights. */
		void setFractionalBits(unsigned bits);

	private:

		friend class nemo::SimulationBackend;
		friend class nemo::mpi::Master;
		friend class nemo::mpi::Worker;

		friend std::ostream& ::operator<<(std::ostream& o, Configuration const&);

		class ConfigurationImpl* m_impl;

		// undefined
		Configuration& operator=(const Configuration&);
};

} // end namespace nemo


#endif
