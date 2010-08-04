#include "devices.hpp"


#include <map>
#include <boost/format.hpp>

#include <nemo/config.h>
#include <nemo/ConfigurationImpl.hpp>

#include "exception.hpp"
#include "FiringOutput.hpp"
#include "kernel.cu_h"

namespace nemo {
	namespace cuda {


typedef std::map<unsigned, cudaDeviceProp> devmap_t;

devmap_t* g_devices = NULL;

unsigned g_defaultDevice = -1;


#ifndef __DEVICE_EMULATION__

bool
deviceSuitable(unsigned device)
{
	cudaDeviceProp prop;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));

	/* 9999.9999 is the 'emulation device' which is always present. Unless the
	 * library was built specifically for emulation mode, this should be
	 * considered an error. */
	if(prop.major == 9999 || prop.minor == 9999) {
		return false;
	}

	/* 1.2 required for shared memory atomics */
	if(prop.major <= 1 && prop.minor < 2) {
		return false;
	}

	return true;
}




unsigned
getDefaultDevice(const devmap_t& devmap)
{
	cudaDeviceProp prop;
	prop.major = 1;
	prop.minor = 2;

	int dev;
	CUDA_SAFE_CALL(cudaChooseDevice(&dev, &prop));

	/* The chose device should already be present in the device map */
	if(devmap.find(dev) == devmap.end()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Internal error: default device not in device map");
	}

	return unsigned(dev);
}

#endif



/* Create and populate a (global) list of suitable devices */
const devmap_t&
enumerateDevices()
{
	if(g_devices == NULL) {

		g_devices = new devmap_t();
#ifdef __DEVICE_EMULATION__
		cudaDeviceProp prop;
		prop.major = 9999;
		prop.minor = 9999;
		int dev;
		CUDA_SAFE_CALL(cudaChooseDevice(&dev, &prop));
		(*g_devices)[0] = prop;
		g_defaultDevice = 0;
#else
		int dcount = 0;
		CUDA_SAFE_CALL(cudaGetDeviceCount(&dcount));
		for(int device = 0; device < dcount; ++device) {
			if(deviceSuitable(device)) {
				cudaDeviceProp prop;
				CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
				(*g_devices)[device] = prop;
			}
		}

		g_defaultDevice = getDefaultDevice(*g_devices);
#endif
	}
	return *g_devices;
}


unsigned
deviceCount()
{
	return enumerateDevices().size();
}


const char*
deviceDescription(unsigned device)
{
	using boost::format;

	const devmap_t& devices = enumerateDevices();
	devmap_t::const_iterator i = devices.find(device);
	if(i == devices.end()) {
		throw nemo::exception(NEMO_CUDA_ERROR, str(format("Invalid device: %u") % device));
	}

	return i->second.name;
}



void
setDevice(unsigned device)
{
	using boost::format;

	const devmap_t& devmap = enumerateDevices();
	if(devmap.find(device) == devmap.end()) {
		throw nemo::exception(NEMO_CUDA_ERROR, str(format("Invalid device: %u") % device));
	}

	int userDev = int(device);
	int existingDev;
	CUDA_SAFE_CALL(cudaGetDevice(&existingDev));
	if(existingDev != userDev) {
		/* If the device has already been set, we'll get an error here */
		CUDA_SAFE_CALL(cudaSetDevice(userDev));
	}
}


void
chooseDevice(ConfigurationImpl& conf, int device)
{
	using boost::format;

	const devmap_t& devmap = enumerateDevices();
	if(deviceCount() == 0) {
		throw nemo::exception(NEMO_CUDA_ERROR, "No CUDA devices available");
	}
	conf.setBackend(NEMO_BACKEND_CUDA);
	if(device < 0) {
		/* Use default */
		conf.setCudaDevice(nemo::cuda::g_defaultDevice);
	} else {
		if(devmap.find(device) == devmap.end()) {
			throw nemo::exception(NEMO_CUDA_ERROR, str(format("Invalid device: %u") % device));
		}
		conf.setCudaDevice(unsigned(device));
	}

	conf.setCudaPartitionSize(MAX_PARTITION_SIZE);
	conf.setCudaFiringBufferLength(FiringOutput::defaultBufferLength());
}


}	}


/* C API */


void
nemo_cuda_choose_device(nemo::ConfigurationImpl* conf, int device)
{
	nemo::cuda::chooseDevice(*conf, device);
}


void
nmoe_cuda_test_configuration(nemo::ConfigurationImpl)
{

}


unsigned
nemo_cuda_device_count()
{
	return nemo::cuda::deviceCount();
}



const char*
nemo_cuda_device_description(unsigned device)
{
	return nemo::cuda::deviceDescription(device);
}
