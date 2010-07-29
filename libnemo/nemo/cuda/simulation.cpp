#include "simulation.hpp"
#include "Simulation.hpp"

#include <nemo/SimulationBackend.hpp>

namespace nemo {
	namespace cuda {

SimulationBackend*
simulation(const NetworkImpl& net, const ConfigurationImpl& conf)
{
	/* We need to select the device before calling the constructor. The
	 * constructor sends data to the device, so we need to know in advance what
	 * device to use. If we call the constructor directly a default device will
	 * be used.  */
	Simulation::selectDevice(conf.cudaDevice());
	return new Simulation(net, conf);
}

}	}
