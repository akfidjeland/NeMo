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

#include <ostream>
#include <vector>

#include <nemo/config.h>
#include <nemo/constants.h>


namespace nemo {
	class Configuration;
}


NEMO_DLL_PUBLIC
std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf);


namespace nemo {

	class Simulation;
	class Network;
	class HardwareConfiguration;

	namespace mpi {
		class Master;
		class Worker;
	}

class NEMO_DLL_PUBLIC Configuration
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

		void setStdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight);

		/*! Specify that the CUDA backend should be used and optionally specify
		 * a desired device. If the (default) device value of -1 is used the
		 * backend will choose the best available device.
		 *
		 * If the cuda backend (and the chosen device) cannot be used for
		 * whatever reason, an exception is raised.
		 *
		 * The device numbering is the numbering used internally by nemo (\see
		 * \a cudaDeviceCount and \a cudaDeviceDescription). This device
		 * numbering may differ from the one provided by the CUDA driver
		 * directly, since nemo ignores any devices it cannot use. */
		void setCudaBackend(int device = -1);

		/*! Specify that the CPU backend should be used and optionally specify
		 * the number of threads to use. If the default thread count of -1 is
		 * used, the backend will choose a sensible value */
		void setCpuBackend(int threadCount = -1);

		backend_t backend() const;

		/*! \return the chosen CUDA device or -1 if CUDA is not the selected
		 * backend. */
		int cudaDevice() const;

		/*! \return the number of threads used by the CPU backend or -1 if CPU
		 * is not the selected backend. */
		int cpuThreadCount() const;


		/*! \return description of the chosen backend */
		const std::string& backendDescription() const;

	private:

		friend NEMO_DLL_PUBLIC Simulation* simulation(const Network& net, const Configuration&);
		friend class nemo::mpi::Master;
		friend class nemo::mpi::Worker;

		friend std::ostream& ::operator<<(std::ostream& o, Configuration const&);

		class ConfigurationImpl* m_impl;

		// undefined
		Configuration& operator=(const Configuration&);

		void setBackendDescription();
};

} // end namespace nemo


#endif
