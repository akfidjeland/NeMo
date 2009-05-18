#include "kernel.h"
#include <cuda_runtime.h>
#include <stdlib.h>



//-----------------------------------------------------------------------------
// DEVICE PROPERTIES
//-----------------------------------------------------------------------------

int
deviceCount()
{
	int count;
	//! \todo error handling
	cudaGetDeviceCount(&count);

	/* Even if there are no actual devices, this function will return 1, which
	 * means that device emulation can be used. We therefore need to check the
	 * major and minor device numbers as well */
	if(count == 1) {
		struct cudaDeviceProp prop; 
		cudaGetDeviceProperties(&prop, 0);
		if(prop.major == 9999 && prop.minor == 9999) {
			count = 0;
		}
	}
	return count;
}



struct cudaDeviceProp*
deviceProperties(int deviceNumber)
{
	struct cudaDeviceProp* prop = 
		(struct cudaDeviceProp*) malloc(sizeof(struct cudaDeviceProp));
	if(prop == NULL) {
		//! \todo error handling
		return prop;
	} 
	
	cudaGetDeviceProperties(prop, deviceNumber); 
	//! \todo handle cuda errors

	return prop;
}


size_t 
totalGlobalMem(struct cudaDeviceProp* prop)
{
	return prop->totalGlobalMem;	
}


size_t 
sharedMemPerBlock(struct cudaDeviceProp* prop)
{
	return prop->sharedMemPerBlock;	
}


int 
regsPerBlock(struct cudaDeviceProp* prop)
{
	return prop->regsPerBlock;	
}


/*
int 
warpSize(struct cudaDeviceProp* prop)
{
	return prop->warpSize;	
}
*/


size_t
memPitch(struct cudaDeviceProp* prop)
{
	return prop->memPitch;	
}



int 
maxThreadsPerBlock(struct cudaDeviceProp* prop)
{
	return prop->maxThreadsPerBlock;	
}



//! \todo maxThreadsDim
//! \todo maxGridSize


size_t 
totalConstMem(struct cudaDeviceProp* prop)
{
	return prop->totalConstMem;	
}



#if 0
int 
major(struct cudaDeviceProp* prop)
{
	return prop->major;	
}



int 
minor(struct cudaDeviceProp* prop)
{
	return prop->minor;	
}
#endif

int
clockRate(struct cudaDeviceProp* prop)
{
	return prop->clockRate;
}
