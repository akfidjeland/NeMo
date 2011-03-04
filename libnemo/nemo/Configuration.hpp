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
#include <nemo/types.h>


namespace nemo {
	class Configuration;
}


NEMO_DLL_PUBLIC
std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf);


namespace nemo {

	class SimulationBackend;
	class Network;
	class ConfigurationImpl;

	namespace mpi {
		class Master;
		class Worker;
	}

class NEMO_DLL_PUBLIC Configuration
{
	public:

		Configuration();

		Configuration(const Configuration&);

		Configuration(const ConfigurationImpl& other, bool ignoreBackendOptions);

		~Configuration();

		/*! Switch on logging and send output to stdout */
		void enableLogging(); 

		void disableLogging();
		bool loggingEnabled() const;

		void setCudaPartitionSize(unsigned ps); 
		unsigned cudaPartitionSize() const;

		void setStdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight);

		/*! Make the synapses write-only
		 *
		 * By default synapse state can be read back at run-time. This may
		 * require setting up data structures of considerable size before
		 * starting the simulation. If the synapse state is not required at
		 * run-time, specify that synapses are write-only in order to save
		 * memory. By default synapses are readable */
		void setWriteOnlySynapses();
		bool writeOnlySynapses() const;

		/*! Specify that the CUDA backend should be used and optionally specify
		 * a desired device. If the (default) device value of -1 is used the
		 * backend will choose the best available device.
		 *
		 * If the cuda backend (and the chosen device) cannot be used for
		 * whatever reason, an exception is raised.
		 *
		 * The device numbering is the numbering used internally by NeMo This
		 * device numbering may differ from the one provided by the CUDA driver
		 * directly, since NeMo ignores any devices it cannot use. */
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
		const char* backendDescription() const;

	private:

		friend SimulationBackend* simulationBackend(const Network&, const Configuration&);
		friend class nemo::mpi::Master;
		friend class nemo::mpi::Worker;

		friend std::ostream& ::operator<<(std::ostream& o, Configuration const&);

		ConfigurationImpl* m_impl;

		// undefined
		Configuration& operator=(const Configuration&);

		void setBackendDescription();
};

} // end namespace nemo


#endif
