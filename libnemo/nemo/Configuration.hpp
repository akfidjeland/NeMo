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

		/*! Specify the number of threads (>= 1) to use for the CPU backend. */
		void setCpuThreadCount(unsigned threads);

		void setCudaPartitionSize(unsigned ps); 
		unsigned cudaPartitionSize() const;

		/*! Set the size of the firing buffer such that it can contain a fixed
		 * number of \a cycles worth of firing data before overflowing. */
		void setCudaFiringBufferLength(unsigned cycles); 
		unsigned cudaFiringBufferLength() const;

		/*! Specify the device which should be used by the CUDA backend when
		 * creating the simulation. The chosen device is not tested here, but
		 * an error may be generated later (when constructing the simulation)
		 * if the chosen device is invalid. There's no need to call this
		 * function; if the device is not specified and the CUDA backend is
		 * used, a suitable device will be chosen */
		void setCudaDevice(int dev);
		int cudaDevice() const;

		void setStdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight);

		/*! The simulation uses a fixed-point format internally for synaptic
		 * weights. Call this method to specify how many fractional bits to
		 * use. If nothing is specified the backend chooses a sensible value
		 * based on the range of weights in the input network. */
		void setFractionalBits(unsigned bits);

		void setBackend(backend_t backend);

		/*! Test whether the configuration is valid, i.e. whether it's possible
		 * to create a simulation based on it. A configuration can be invalid
		 * for a number of reasons including. Use \a descr to check reason for
		 * configuration being invalid. Returns true/ok if the test passes. */
		bool test();

		/*! \return description of the backend used or any error */
		const std::string& backendDescription() const;

	private:

		friend NEMO_DLL_PUBLIC Simulation* simulation(const Network& net, Configuration& conf);
		friend class nemo::mpi::Master;
		friend class nemo::mpi::Worker;

		friend std::ostream& ::operator<<(std::ostream& o, Configuration const&);

		class ConfigurationImpl* m_impl;

		// undefined
		Configuration& operator=(const Configuration&);
};

} // end namespace nemo


#endif
