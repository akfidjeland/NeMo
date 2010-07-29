#ifndef NEMO_CUDA_SIMULATION_FACTORY_HPP
#define NEMO_CUDA_SIMULATION_FACTORY_HPP

#include <nemo/config.h>

namespace nemo {

	class NetworkImpl;
	class ConfigurationImpl;
	class SimulationBackend;

	namespace cuda {

NEMO_CUDA_DLL_PUBLIC
SimulationBackend*
simulation(const NetworkImpl& net, const ConfigurationImpl& conf);

}	}

#endif
