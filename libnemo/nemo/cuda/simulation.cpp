#include "simulation.hpp"
#include "Simulation.hpp"


nemo::SimulationBackend*
nemo_cuda_simulation(const nemo::NetworkImpl* net, nemo::ConfigurationImpl* conf)
{
	return nemo::cuda::simulation(*net, *conf);
}


void
nemo_cuda_test_simulation(nemo::ConfigurationImpl* conf)
{
	nemo::cuda::Simulation::test(*conf);
}


namespace nemo {
	namespace cuda {

SimulationBackend*
simulation(const NetworkImpl& net, ConfigurationImpl& conf)
{
	/* We need to select the device before calling the constructor. The
	 * constructor sends data to the device, so we need to know in advance what
	 * device to use. If we call the constructor directly a default device will
	 * be used.  */
	Simulation::test(conf);
	return new Simulation(net, conf);
}

}	}
