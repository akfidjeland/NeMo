#ifndef NEMO_CUDA_DEVICES_HPP
#define NEMO_CUDA_DEVICES_HPP

#include <nemo/config.h>

namespace nemo {
	class ConfigurationImpl;
}

extern "C" {

typedef void nemo_cuda_choose_device_t(nemo::ConfigurationImpl*, int);


/* Choose the CUDA device to use and fill in the relevant field in the
 * configuration object. If dev = -1, have the backend choose a suitable
 * device, otherwise check that the user-selected device is a valid choice */
NEMO_CUDA_DLL_PUBLIC
void
nemo_cuda_choose_device(nemo::ConfigurationImpl*, int dev);



typedef void nemo_cuda_test_device_t(unsigned device);

NEMO_CUDA_DLL_PUBLIC
void
nemo_cuda_test_device(unsigned device);



typedef unsigned nemo_cuda_device_count_t(void);

NEMO_CUDA_DLL_PUBLIC
unsigned
nemo_cuda_device_count(void);



typedef const char* nemo_cuda_device_description_t(unsigned device);

NEMO_CUDA_DLL_PUBLIC
const char*
nemo_cuda_device_description(unsigned device);

}


namespace nemo {
	namespace cuda {

/* Commit to using a specific device for this *thread*. If called multiple
 * times with different devices chose, this *may* raise an exception */
void setDevice(unsigned device);

}	}


#endif
