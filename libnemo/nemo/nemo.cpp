/* This file contains factory methods only, both for the public and the
 * internal API. Most functionality is found in the respective C++ classes and
 * in the C API wrapper file nemo_c.cpp */

#include <nemo/config.h>

#ifdef NEMO_CUDA_DYNAMIC_LOADING
#include <ltdl.h>
#endif

#include <boost/format.hpp>


#include <nemo/internals.hpp>
#include <nemo/exception.hpp>

#ifdef NEMO_CUDA_ENABLED
#include <nemo/cuda/simulation.hpp>
#endif
#include <nemo/cpu/Simulation.hpp>

namespace nemo {

#ifdef NEMO_CUDA_DYNAMIC_LOADING

lt_dlhandle libcuda = NULL;

void
unloadCudaLibrary()
{
	if(libcuda != NULL) {
		lt_dlclose(libcuda);
		lt_dlexit(); // since the cuda backend is the only dynamically opened library
	}
}


lt_dlhandle
loadCudaLibrary()
{
	using boost::format;

	if(libcuda == NULL) {
		if(lt_dlinit() != 0) {
			throw nemo::exception(NEMO_DL_ERROR, lt_dlerror());
		}
		libcuda = lt_dlopenext("libnemo_cuda");
		if(libcuda == NULL) {
			throw nemo::exception(NEMO_DL_ERROR, str(format("failed to open nemo_cuda library: %s") % lt_dlerror()));
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
	lt_dlhandle hdl = loadCudaLibrary();
	nemo_cuda_simulation_t* ctor = (nemo_cuda_simulation_t*) lt_dlsym(hdl, "nemo_cuda_simulation");
	if(ctor == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, lt_dlerror());
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
	lt_dlhandle hdl = loadCudaLibrary();
	nemo_cuda_test_simulation_t* test = (nemo_cuda_test_simulation_t*) lt_dlsym(hdl, "nemo_cuda_test_simulation");
	if(test == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, lt_dlerror());
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

	if(conf.backend() == NEMO_BACKEND_CUDA) {
		try {
			testCuda(conf);
		} catch (nemo::exception& e) {
			conf.setBackendDescription(str(format("Cannot simulate on CUDA device: %s") % e.what()));
			valid = false;
		}
	} else if(conf.backend() == NEMO_BACKEND_UNSPECIFIED) {
		try {
			testCuda(conf);
		} catch(nemo::exception&) {
			/* The CUDA backend does not work for some reason. However, the
			 * user did not specify what backend to use, so just go ahead and
			 * use the CPU backend instead. */
			cpu::Simulation::test(conf);
		}
	} else if(conf.backend() == NEMO_BACKEND_CPU) {
			cpu::Simulation::test(conf);
	} else {
		throw nemo::exception(NEMO_LOGIC_ERROR, "unknown backend in configuration");
	}
	return true;
}

}
