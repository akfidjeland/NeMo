/* This file contains factory methods only, both for the public and the
 * internal API. Most functionality is found in the respective C++ classes and
 * in the C API wrapper file nemo_c.cpp */

#include <nemo/config.h>

#ifdef NEMO_CUDA_DYNAMIC_LOADING
#include "dyn_load.hpp"
#endif

#include <boost/format.hpp>


#include <nemo/internals.hpp>
#include <nemo/exception.hpp>

#ifdef NEMO_CUDA_ENABLED
#include <nemo/cuda/create_simulation.hpp>
#endif
#include <nemo/cpu/Simulation.hpp>

namespace nemo {

#ifdef NEMO_CUDA_DYNAMIC_LOADING

dl_handle libcuda = NULL;

void
unloadCudaLibrary()
{
	if(libcuda != NULL) {
		dl_unload(libcuda);
		dl_exit(); // since the cuda backend is the only dynamically opened library
	}
}


dl_handle
loadCudaLibrary()
{
	using boost::format;

	if(libcuda == NULL) {
		if(!dl_init()) {
			throw nemo::exception(NEMO_DL_ERROR, dl_error());
		}
		libcuda = dl_load(LIB_NAME("nemo_cuda"));
		if(libcuda == NULL) {
			throw nemo::exception(NEMO_DL_ERROR, str(format("failed to open nemo_cuda library: %s") % dl_error()));
		}
		atexit(unloadCudaLibrary);
	}
	return libcuda;
}


#endif


SimulationBackend*
cudaSimulation(const NetworkImpl& net, ConfigurationImpl& conf)
{
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	dl_handle hdl = loadCudaLibrary();
	nemo_cuda_simulation_t* ctor = (nemo_cuda_simulation_t*) dl_sym(hdl, "nemo_cuda_simulation");
	if(ctor == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, dl_error());
	}
	return ctor(&net, &conf);
#else
	return cuda::simulation(net, conf);
#endif
}



/* Sometimes using the slightly lower-level interface provided by NetworkImpl
 * makes sense (see e.g. nemo::mpi::Worker), so provide an overload of 'create'
 * that takes such an object directly. */
SimulationBackend*
simulationBackend(const NetworkImpl& net, ConfigurationImpl& conf)
{
	if(net.neuronCount() == 0) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot create simulation from empty network");
		return NULL;
	}

	switch(conf.backend()) {
#ifdef NEMO_CUDA_ENABLED
		case NEMO_BACKEND_UNSPECIFIED:
			try {
				return cudaSimulation(net, conf);
			} catch(...) {
				return new cpu::Simulation(net, conf);
			}
		case NEMO_BACKEND_CUDA:
			return cudaSimulation(net, conf);
#else
		case NEMO_BACKEND_CUDA:
			throw nemo::exception(NEMO_API_UNSUPPORTED,
					"nemo was compiled without Cuda support. Cannot create simulation");
		case NEMO_BACKEND_UNSPECIFIED:
#endif
		case NEMO_BACKEND_CPU:
			return new cpu::Simulation(net, conf);
		default :
			throw nemo::exception(NEMO_LOGIC_ERROR, "unknown backend in configuration");
	}
}


Simulation*
simulation(const Network& net, Configuration& conf)
{
	return dynamic_cast<Simulation*>(simulationBackend(*net.m_impl, *conf.m_impl));
}



/* Check that
 *
 * 1. we can load the CUDA library
 * 2. that the simulation parameters check out
 *
 * As a side effect, fill in missing relevant fields in conf and add a backend
 * description string.
 *
 * Errors are signaled via exceptions.
 */
void
testCuda(ConfigurationImpl& conf)
{
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	dl_handle hdl = loadCudaLibrary();
	nemo_cuda_test_simulation_t* test = (nemo_cuda_test_simulation_t*) dl_sym(hdl, "nemo_cuda_test_simulation");
	if(test == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, dl_error());
	}
	test(&conf);
#else
	cuda::testSimulation(conf);
#endif
}



bool
testBackend(ConfigurationImpl& conf)
{
	using boost::format;
	bool valid = true;

	try {

	if(conf.backend() == NEMO_BACKEND_CUDA) {
		try {
			testCuda(conf);
		} catch (std::exception& e) {
			conf.setBackendDescription(str(format("Cannot simulate on CUDA device: %s") % e.what()));
			valid = false;
		}
	} else if(conf.backend() == NEMO_BACKEND_UNSPECIFIED) {
		try {
			testCuda(conf);
		} catch(std::exception&) {
			/* The CUDA backend does not work for some reason. However, the
			 * user did not specify what backend to use, so just go ahead and
			 * use the CPU backend instead. */
			cpu::Simulation::test(conf);
		}
	} else if(conf.backend() == NEMO_BACKEND_CPU) {
			cpu::Simulation::test(conf);
	} else {
		conf.setBackendDescription("Unknown backend specified in configuration");
		valid = false;
	}

	} catch(...) {
		conf.setBackendDescription("An unkown exception was raised when testing the backend");
		valid = false;
	}
	return valid;
}

}
