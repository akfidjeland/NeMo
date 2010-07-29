#ifndef NEMO_CUDA_SIMULATION_FACTORY_HPP
#define NEMO_CUDA_SIMULATION_FACTORY_HPP

#include <nemo/config.h>

namespace nemo {
	class NetworkImpl;
	class ConfigurationImpl;
	class SimulationBackend;
}

extern "C" {

typedef nemo::SimulationBackend* nemo_cuda_simulation_t(const nemo::NetworkImpl* net, const nemo::ConfigurationImpl* conf);

NEMO_CUDA_DLL_PUBLIC
nemo::SimulationBackend*
nemo_cuda_simulation(const nemo::NetworkImpl* net, const nemo::ConfigurationImpl* conf);
}


namespace nemo {
	namespace cuda {

NEMO_CUDA_DLL_PUBLIC
SimulationBackend*
simulation(const NetworkImpl& net, const ConfigurationImpl& conf);

}	}

#endif
