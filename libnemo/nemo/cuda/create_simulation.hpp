#ifndef NEMO_CUDA_SIMULATION_FACTORY_HPP
#define NEMO_CUDA_SIMULATION_FACTORY_HPP

#include <nemo/config.h>

namespace nemo {
	namespace network {
		class NetworkImpl;
	}
	class ConfigurationImpl;
	class SimulationBackend;
}

extern "C" {

typedef nemo::SimulationBackend* cuda_simulation_t(const nemo::network::NetworkImpl*, const nemo::ConfigurationImpl*);

NEMO_CUDA_DLL_PUBLIC
nemo::SimulationBackend*
cuda_simulation(const nemo::network::NetworkImpl* net, const nemo::ConfigurationImpl* conf);

}

#endif
