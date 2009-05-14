//! \file Simulation.cpp

#include "Simulation.hpp"
#include "DeviceMemory.hpp"
#include "izhikevich.h"
#include "izhikevich_kernel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <assert.h>





Simulation::Simulation() :
	//! \todo default to whole network
	m_clusterProbe(0),
	m_neuronProbe(ALL)
{}



int
Simulation::addCluster(const Cluster& c)
{
	clusters.push_back(c);
	return clusters.size()-1;
}



/* allocate local memory structures -- device memory only created in run() */
int 
Simulation::addCluster(int n, 
		const float* v,
		const float* u,
		const float* a,
		const float* b,
		const float* c,
		const float* d,
		const float* connectionStrength,
		bool hasExternalCurrent,
		bool hasExternalFiring)
{
	Cluster cluster(n, v, u, a, b, c, d, connectionStrength);

	//! \todo deal with delay here!

	if(hasExternalCurrent)
		cluster.enableExternalCurrent();
	if(hasExternalFiring)
		cluster.enableExternalFiring();
	clusters.push_back(cluster);
	//! \todo deal with errors here!
	return clusters.size() - 1;
}



//=============================================================================
// Simulation statistics
//=============================================================================



/*! \return the number of firing neurons among the first \a n neurons in the
 * firing vector */
int
presynapticSpikes(char* firing, int n)
{
	assert(firing != NULL);
	//! \todo tidy this. Remove ifdef globally, or better, deal with packed bits here
#ifdef PACK_SPIKE_BITS
#error "Packed spike bits not implemented"
#else
	int count = 0;
	for( int i=0; i < n; ++i ){
		count += firing[i] ? 1 : 0;
	}
	return count;
#endif
}



/*! \return  proportion of the first \a n neurons in the firing vector which
 * fired */
float
firingRatio(char* firing, int n)
{
	assert(firing != NULL);
	return float(presynapticSpikes(firing, n))/float(n);
}



uint64_t
postsynapticSpikes(char* firing, int n, const Cluster& cluster) 
{
	uint64_t count = 0;
	for(int pre=0; pre < n; ++pre){
		if(firing[pre]) {
			count += cluster.postsynapticCount(pre);
		}
	}
	return count;
}



/* Print memory accesses (load, store) seprated into main, fire, and integrate
 * steps */
void
memoryAccesses(FILE* out, char* firing, int n, int cycle)
{
	int f = presynapticSpikes(firing, n);
	int loadMain = 2*n + n /4;
	int loadFire = 2 + 2*f;
	int loadIntegrate = n + n*f;
	int load = loadMain + loadFire + loadIntegrate;
	int store = 2*n + n/4;

	fprintf(out, "cycle=%d\tneurons=%d\tfiring=%d\ttotal=%d\tstore=%d\tload=%d\tload_integrate=%d\n",
			cycle, n, f, store+load, store, load, loadIntegrate);
}



//=============================================================================
// Probing
//=============================================================================


void
Simulation::setClusterProbe(int cluster)
{
	m_clusterProbe = cluster;	
}



void
Simulation::setNeuronProbe(int neuron)
{
	m_neuronProbe = neuron;
}



//=============================================================================
// Firing vector
//=============================================================================


/* Currently, the firing handlers deal with an array of chars. However, the
 * firing information that is read back is a vector of densely packed ints.
 * The LSb is the most recent spike. */
void
unpackFiring(int* dense, char* sparse, int length, int delay)
{
	for( int i=0; i < length; ++i ){
		uint32_t bits = dense[i]; 
		sparse[i] = (bits >> (delay-1)) & 0x1; 
	}
}



/*! \param sparse
 * 		Array of firing data a single char per neuron. Non-0 means firing.
 * 	\param dense
 * 		Densely packed firing data, 1 bit per neuron.
 *  \param length
 *  	Number of neurons
 *  \return
 *  	True if there is at least on stimulus this cycle
 */
bool
packFiring(std::vector<char>& sparse, uint32_t* dense, int length)
{
	bool haveFiring = false;
	for(int chunk=0; chunk < length/32 + length%32 ? 1 : 0; ++chunk) {
		uint32_t bits = 0;
	 	for(int nn=chunk*32; nn < (chunk+1)*32; ++nn) {
			bits >>= 1;
			if(nn < length) {
				bits |= (sparse[nn] ? 0x1 << 31 : 0x0);
			}
 		}
 		dense[chunk] = bits;
		haveFiring |= bits;
	}
	return haveFiring;
}



//=============================================================================
// Running simulation
//=============================================================================


SimStatus
Simulation::run(int totalCycles, 
		int updatesPerInvocation,
		int reportFlags, 
		float currentScaling,
		void(*currentStimulus)(float*, int, int),
		void(*firingStimulus)(char*, int, int),
		HandlerError(*firingHandler)(FILE*, char*, int, int, uchar),
		FILE* firingFile,
		HandlerError(*vHandler)(FILE*, float*, int),
		FILE* vFile,
		int outputStart,
		int outputDuration,
		//void(*uHandler)(FILE*, float*, int)=NULL);
		bool forceDense)
{
	KernelError kernelError = KERNEL_OK;

	if(clusters.size() < 1){
		std::cerr << "No neuronal clusters specified. Aborting\n";
		return SIM_ERROR;
	}

	DeviceMemory mem(clusters, forceDense, reportFlags & REPORT_MEMORY);
	kernelError = configureClusters(
			mem.hasExternalCurrent(),
			mem.hasExternalFiring(),
			mem.maxColumnIndex());
	if(kernelError != KERNEL_OK) {
		std::cerr << "Error in configuring clusters. Aborting\n";
		return SIM_ERROR;
	}

	int* firing = NULL;
	//! \todo use std::vector for memory which is not page-locked
	char* firingSparse = NULL;
	if(firingHandler != NULL || reportFlags & REPORT_FIRING || reportFlags & REPORT_POSTSYNAPTIC){ 
		cudaMallocHost((void**)&firing, mem.n*sizeof(int));
		firingSparse = (char*) calloc(mem.n, sizeof(char));
	}

	float* v = NULL;
	if(vHandler != NULL) {
		cudaMallocHost((void**)&v, mem.n*sizeof(float));
	}

	//! \todo deal with external inputs in a more sensible manner
	float* extI = NULL;
	if(currentStimulus != NULL)
		cudaMallocHost((void**)&extI, mem.n*sizeof(float));

	std::vector<char> extFiring(mem.n, 0);
	uint32_t* extFiringPacked = NULL;
	if(firingStimulus != NULL) {
		cudaMallocHost((void**)&extFiringPacked, mem.pitch1);
		//! \todo clear this memory
	}

	/* Running average of number of spikes per cycle */
	float ratioAcc = 0.0f;
	int ratioCount = 0;
	uint64_t firingAcc = 0;
	uint64_t postsynapticAcc = 0;

	clock_t startTime = clock();
	int kernelInvocations = 0;
	int currentCycle = 0;
	HandlerError handlerError = HANDLER_OK;

	int device;
	cudaGetDevice(&device);
	cudaDeviceProp deviceProperties; 
	cudaGetDeviceProperties(&deviceProperties, device);

	while(currentCycle < totalCycles) {

		if(currentStimulus != NULL)
			currentStimulus(extI, mem.n, currentCycle);

		bool haveExtFiring = false; 
		if(firingStimulus != NULL) {
			firingStimulus(&extFiring[0], mem.n, currentCycle);
			haveExtFiring = packFiring(extFiring, extFiringPacked, mem.n);
		}

		kernelError = step(&deviceProperties, 
				currentCycle,
				updatesPerInvocation, &mem, 
				currentScaling,
				extI, 
				haveExtFiring ? extFiringPacked : 0, 
				firing, v, m_clusterProbe);

		if(kernelError != KERNEL_OK) {
			break;
		}

		currentCycle += updatesPerInvocation;
		++kernelInvocations;

		for(int d=updatesPerInvocation; d >= 1; --d){

			int cycle = currentCycle - d;
			bool final = cycle == totalCycles-1;

			if( cycle >= outputStart 
					&& ( outputDuration <= 0 
						|| cycle <= outputStart+outputDuration) ) {

				if(firing && firingSparse) {
					unpackFiring(firing, firingSparse, mem.n, d);
				}

				if(firingHandler != NULL){
					char* start = m_neuronProbe == ALL ? firingSparse : firingSparse+m_neuronProbe;
					int len = m_neuronProbe == ALL ? mem.n : 1;
					handlerError = firingHandler(firingFile, start, len, cycle, final); 
					if(handlerError != HANDLER_OK) {
						break;
					}
				}

				if(vHandler != NULL){
					//! \todo deal with neuron probe properly
					float* start = m_neuronProbe == ALL ? v : v+m_neuronProbe;
					int len = m_neuronProbe == ALL ? mem.n : 1;
					handlerError = vHandler(vFile, start, len);
					if(handlerError != HANDLER_OK) {
						break;
					}
				}

				if(reportFlags & REPORT_FIRING) {
					ratioAcc += firingRatio(firingSparse, mem.n);
					firingAcc += presynapticSpikes(firingSparse, mem.n);
					ratioCount++;
				}

#if 0
				if(reportFlags & REPORT_POSTSYNAPTIC) {
					//! \todo deal with probing of other clusters here!
					postsynapticAcc += postsynapticSpikes(firingSparse, mem.n, clusters.front());		
				}
#endif

#if 0
				if(reportFlags & REPORT_MEMORY) {
					//! \todo don't mess up stdout
					//! \todo print this to file
					memoryAccesses(stdout, firingSparse, mem.n, cycle);
				}
#endif
			}

			if(final) {
				break;
			}
		}
	}

	//! \todo move to separate function
	if(kernelError == KERNEL_OK && handlerError == HANDLER_OK && reportFlags) {
		printf("--------------------------------------------------------------------------------\n");
		double elapsed = ( clock() - startTime ) / double(CLOCKS_PER_SEC);
		if(reportFlags & REPORT_TIMING) {
			int neurons = mem.n*mem.clusterCount();
			printf("Simulated %d neurons (%d clusters) for %d cycles in %fs\n",
					neurons, mem.clusterCount(), totalCycles, elapsed);
			std::cout << "Kernel invocations\n"
			          << "\ttotal: " << kernelInvocations << "\n"
					  << "\tper second: " << kernelInvocations / elapsed << std::endl;
			printf("updates/s: %.0f\n", double(totalCycles*neurons) / elapsed);
			printf("updates/ms: %.0f\n", double(totalCycles*neurons) / (elapsed*1000));
		}

		if(reportFlags & REPORT_FIRING) {
			std::cout << "Presynaptic spikes\n"
			          << "\ttotal: " << firingAcc << "\n"
					  << "\tper second: " << firingAcc / elapsed << "\n"
					  << "\tper cycle: " << firingAcc / (elapsed*1000) << std::endl;
			//! \todo for large values of firingAcc this will probably cause overflow
			printf("Firing ratio: %3f\n", ratioAcc/float(ratioCount) );
		}

		if(reportFlags & REPORT_POSTSYNAPTIC) {
			//! \todo REPORT_POSTSYNAPTIC -> REPORT_FIRING
			//! \todo use cout throughout, rather than printf
			std::cout << "Postsynaptic spikes\n"
			          << "\ttotal: " << postsynapticAcc << "\n"
			          << "\tper second: " << postsynapticAcc / elapsed << "\n"
			          << "\tper cycle: " << postsynapticAcc / (elapsed*1000) << "\n"
					  << "\tper presyaptic: " << postsynapticAcc / firingAcc << std::endl;
			
		}
		printf("--------------------------------------------------------------------------------\n");
	}

	//if(extFiring)
		//cudaFreeHost(extFiring);
	if(extFiringPacked)
		cudaFreeHost(extFiringPacked);

	if(extI)
		cudaFreeHost(extI);

	if(v) 
		cudaFreeHost(v);

	if(firing) 
		cudaFreeHost(firing);

	if(firingSparse)
		free(firingSparse);

	if(kernelError == KERNEL_OK && handlerError == HANDLER_OK) {
		return SIM_OK;
	} else {
		return SIM_ERROR;
	}
}



int
Simulation::maxClusterSize() const
{
	return (max_element(clusters.begin(), clusters.end()))->n;
}




//=============================================================================
// C-based API
//=============================================================================


Simulation* 
nsimGetHandle()
{
	return new Simulation();
}



void 
nsimFreeHandle(Simulation* sim)
{
	delete sim;
}



int 
nsimAddCluster(Simulation* sim, Cluster* cluster)
{
	return sim->addCluster(*cluster);
}



#if 0
int 
nsimAddCluster(CUDA_SIM_HANDLE sim, int n, 
	float* v, float* u, 
	float* a, float* b, float* c, float* d, 
	float* weights)
{
	return sim->addCluster(n, v, u, a, b, c, d, weights);
}
#endif



SimStatus
nsimRun(CUDA_NSIM_HANDLE sim,
		int simCycles, 
		int updatesPerInvocation,
		int reportFlags, 
		float currentScaling,
		void(*currentStimulus)(float*, int, int),
		void(*firingStimulus)(char*, int, int),
		HandlerError(*firingHandler)(FILE*, char*, int, int, uchar),
		FILE* firingFile,
		HandlerError(*vHandler)(FILE*, float*, int),
		FILE* vFile)
		//void(*uHandler)(FILE*, float*, int)=NULL);
{
	return sim->run(simCycles, updatesPerInvocation, 
			reportFlags, 
			currentScaling,
			currentStimulus, 
			firingStimulus,
			firingHandler, firingFile,
			vHandler, vFile);
}
