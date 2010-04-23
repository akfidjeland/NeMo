/* This file contains factory methods only, both for the public and the
 * internal API. Most functionality is found in the respective C++ classes and
 * in the C API wrapper file nemo_c.cpp */

#include <nemo/config.h>
#include <nemo/internals.hpp>
#include <nemo/exception.hpp>

#include <nemo/cuda/CudaSimulation.hpp>
#include <nemo/cpu/Simulation.hpp>

namespace nemo {

/* Sometimes using the slightly lower-level interface provided by NetworkImpl
 * makes sense (see e.g. nemo::mpi::Worker), so provide an overload of 'create'
 * that takes such an object directly. */
SimulationBackend*
simulationBackend(const NetworkImpl& net, const ConfigurationImpl& conf)
{
	if(net.neuronCount() == 0) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot create simulation from empty network");
		return NULL;
	}

	int dev;
	switch(conf.backend()) {
		case NEMO_BACKEND_UNSPECIFIED:
		case NEMO_BACKEND_CUDA:
			dev = cuda::Simulation::selectDevice();
			if(dev == -1) {
				throw nemo::exception(NEMO_CUDA_ERROR, "Failed to create simulation");
			}
			return new cuda::Simulation(net, conf);
		case NEMO_BACKEND_CPU:
			return new cpu::Simulation(net, conf);
		default :
			throw nemo::exception(NEMO_LOGIC_ERROR, "unknown backend in configuration");
	}
}


Simulation*
simulation(const Network& net, const Configuration& conf)
{
	return dynamic_cast<Simulation*>(simulationBackend(*net.m_impl, *conf.m_impl));
}

}
