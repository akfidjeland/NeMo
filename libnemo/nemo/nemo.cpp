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
#include <nemo/cuda/devices.hpp>
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




unsigned
cudaDeviceCount()
{
#ifdef NEMO_CUDA_ENABLED
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	dl_handle hdl  = loadCudaLibrary();
	nemo_cuda_device_count_t* fn = (nemo_cuda_device_count_t*) dl_sym(hdl, "nemo_cuda_device_count");
	if(fn == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, dl_error());
	}
	return fn();
#else
	return nemo_cuda_device_count();
#endif
#else // NEMO_CUDA_ENABLED
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"libnemo compiled without CUDA support");
#endif // NEMO_CUDA_ENABLED
}



/* Throws on error */
const char*
cudaDeviceDescription(unsigned device)
{
#ifdef NEMO_CUDA_ENABLED
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	dl_handle hdl  = loadCudaLibrary();
	nemo_cuda_device_description_t* fn =
		(nemo_cuda_device_description_t*) dl_sym(hdl, "nemo_cuda_device_description");
	if(fn == NULL) {
		throw nemo::exception(NEMO_DL_ERROR, dl_error());
	}
	return fn(device);
#else
	return cuda_device_description(device);
#endif
#else // NEMO_CUDA_ENABLED
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"libnemo compiled without CUDA support");
#endif // NEMO_CUDA_ENABLED
}


#ifdef NEMO_CUDA_ENABLED

SimulationBackend*
cudaSimulation(const NetworkImpl& net, const ConfigurationImpl& conf)
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

#endif


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

	switch(conf.backend()) {
#ifdef NEMO_CUDA_ENABLED
		case NEMO_BACKEND_CUDA:
			return cudaSimulation(net, conf);
#else
		case NEMO_BACKEND_CUDA:
			throw nemo::exception(NEMO_API_UNSUPPORTED,
					"nemo was compiled without Cuda support. Cannot create simulation");
#endif
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




/* Set the default CUDA device if possible. Throws if anything goes wrong or if
 * there are no suitable devices. If device is -1, have the backend choose a
 * device. Otherwise, try to use the device provided by the user.  */
void
setCudaDeviceConfiguration(nemo::ConfigurationImpl& conf, int device)
{
#ifdef NEMO_CUDA_ENABLED
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	dl_handle hdl = loadCudaLibrary();
	nemo_cuda_choose_device_t* fn =
		(nemo_cuda_choose_device_t*) dl_sym(hdl, "nemo_cuda_choose_device");
	fn(&conf, device);
#else
	nemo_cuda_choose_device(&conf, device);
#endif
#else // NEMO_CUDA_ENABLED
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"libnemo compiled without CUDA support");
#endif // NEMO_CUDA_ENABLED
}




void
setDefaultHardware(nemo::ConfigurationImpl& conf)
{
#ifdef NEMO_CUDA_ENABLED
	try {
		setCudaDeviceConfiguration(conf, -1);
	} catch(...) {
		cpu::chooseHardwareConfiguration(conf);
	}
#else
		cpu::chooseHardwareConfiguration(conf);
#endif
}



const char*
version()
{
	return NEMO_VERSION;
}


}
