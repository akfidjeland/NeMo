/* This file contains factory methods only, both for the public and the
 * internal API. Most functionality is found in the respective C++ classes and
 * in the C API wrapper file nemo_c.cpp */

#include <nemo_config.h>

#include "nemo_internal.hpp"
#include "exception.hpp"

#include <CudaSimulation.hpp>

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
	int dev = cuda::Simulation::selectDevice();
	if(dev == -1) {
		throw nemo::exception(NEMO_CUDA_ERROR, "Failed to create simulation");
	}
	return new cuda::Simulation(net, conf);
}


Simulation*
simulation(const Network& net, const Configuration& conf)
{	
	return dynamic_cast<Simulation*>(simulationBackend(*net.m_impl, *conf.m_impl));
}

}
