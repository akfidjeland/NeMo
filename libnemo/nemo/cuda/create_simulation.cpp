#include "create_simulation.hpp"
#include "Simulation.hpp"
#include "devices.hpp"



namespace nemo {
	namespace cuda {

SimulationBackend*
simulation(const NetworkImpl& net, const ConfigurationImpl& conf)
{
	/* We need to select the device before calling the constructor. The
	 * constructor sends data to the device, so we need to know in advance what
	 * device to use. If we call the constructor directly a default device will
	 * be used.  */
	setDevice(conf.cudaDevice());
	return new Simulation(net, conf);
}

}	}

nemo::SimulationBackend*
nemo_cuda_simulation(const nemo::NetworkImpl* net, const nemo::ConfigurationImpl* conf)
{
	return nemo::cuda::simulation(*net, *conf);
}
