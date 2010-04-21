/* 
 * Toroidal network where the torus is constructed from a variable number of
 * 32x32-sized patches of neurons. The connectivity is such that the distance
 * between pre- and postsynaptic neurons is normally distributed (2D euclidian
 * distance along the torus surface) with the tails of the distributions capped
 * at 20x32.  Conductance delays are linearly dependent on distance and ranges
 * from 1ms to 20ms.
 *
 * This network shows usage of libnemo and can be used for benchmarking
 * purposes.
 *
 * Author: Andreas K. Fidjelnad <andreas.fidjeland@imperial.ac.uk>
 * Date: March 2010
 */ 

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/random.hpp>

#include <nemo.hpp>


#define PATCH_WIDTH 32
#define PATCH_HEIGHT 32
#define PATCH_SIZE ((PATCH_WIDTH) * (PATCH_HEIGHT))

#define MAX_DELAY 20U

#define PI 3.14159265358979323846264338327

typedef unsigned char uchar;

/* Random number generators */
typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::normal_distribution<double> > grng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;

/* Global neuron index and distance */
typedef std::pair<unsigned, double> target_t;




/* Return global neuron index given location on torus */
unsigned
neuronIndex(unsigned patch, unsigned x, unsigned y)
{
	assert(x >= 0);
	assert(x < PATCH_WIDTH);
	assert(y >= 0);
	assert(y < PATCH_HEIGHT);
	return patch * PATCH_SIZE + y * PATCH_WIDTH + x; 
}



void
addExcitatoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float a = 0.02f;
	float b = 0.2f;
	float r1 = float(param());
	float r2 = float(param());
	float c = v + 15.0f * r1 * r1;
	float d = 8.0f - 6.0f * r2 * r2;
	float u = b * v;
	float sigma = 5.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
}



void
addInhibitoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float r1 = float(param());
	float a = 0.02f + 0.08f * r1;
	float r2 = float(param());
	float b = 0.25f - 0.05 * r2;
	float c = v; 
	float d = 2.0f;
	float u = b * v;
	float sigma = 2.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
}




/* Round to nearest integer away from zero */
inline
double
round(double r) {
	return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
	//return (r > 0.0) ? ceil(r) : floor(r);
}




target_t
targetNeuron(
		unsigned sourcePartition,
		unsigned sourceX,
		unsigned sourceY,
		unsigned pcount,
		grng_t& distance,
		urng_t& angle)
{
	//! \todo should we set an upper limit to the distance? 
	/* Make sure we don't connect back to self with (near) 0-distance connection.
	 * Perhaps better to simply reject very short distances? */
	double dist = 1.0 + abs(distance());
	double theta = angle();

	double distX = dist * cos(theta);
	double distY = dist * sin(theta);

	/* global x,y-coordinates */
	int globalX = sourcePartition * PATCH_WIDTH + sourceX + round(distX);
	int globalY = sourceY + round(distY);

	int targetY = globalY % PATCH_HEIGHT;
	if(targetY < 0)
		targetY += PATCH_HEIGHT;

	int torusSize = PATCH_WIDTH * pcount;
	globalX = globalX % torusSize;
	if(globalX < 0)
		globalX += torusSize;

	/* We only cross partition boundaries in the X direction */
	// deal with negative numbers here
	int targetPatch = globalX / PATCH_WIDTH;
	int targetX = globalX % PATCH_WIDTH;

	/* Don't connect to self unless we wrap around torus */
	assert(!(targetX == sourceX && targetY == sourceY && dist < PATCH_HEIGHT));

	return std::make_pair<unsigned, double>(neuronIndex(targetPatch, targetX, targetY), dist);
}




unsigned
delay(unsigned distance)
{
	if(distance > MAX_DELAY*PATCH_WIDTH) {
		return MAX_DELAY;
	} else {
		unsigned d = 1 + distance / PATCH_WIDTH;
		assert(d <= MAX_DELAY);
		return d;
	}
}



void
addExcitatorySynapses(
		nemo::Network* net,
		unsigned patch, unsigned x, unsigned y,
		unsigned pcount, unsigned m,
		bool stdp,
		grng_t& distance,
		urng_t& angle,
		urng_t& rweight)
{
	std::vector<unsigned> targets(m, 0U);
	std::vector<unsigned> delays(m, 0U);
	std::vector<float> weights(m, 0.0f);
	std::vector<uchar> isPlastic(m, stdp ? 1 : 0);

	for(unsigned sidx = 0; sidx < m; ++sidx) {
		//! \todo add dependence of delay on distance
		target_t target = targetNeuron(patch, x, y, pcount, distance, angle);

		targets.at(sidx) = target.first;
		weights.at(sidx) = 0.5 * rweight();
		delays.at(sidx) = delay(target.second);
		//std::cout << neuronIndex(patch, x, y) << " -> " << target.first << " d=" << target.second << "\n";
	}

	net->addSynapses(neuronIndex(patch, x, y),
			targets, delays, weights, isPlastic);
}


void
addInhibitorySynapses(
		nemo::Network* net,
		unsigned patch, unsigned x, unsigned y,
		unsigned pcount, unsigned m,
		bool stdp,
		grng_t& distance,
		urng_t& angle,
		urng_t& rweight)
{
	std::vector<unsigned> targets(m, 0);
	std::vector<unsigned> delays(m, 0);
	std::vector<float> weights(m, 0.0f);
	std::vector<uchar> isPlastic(m, stdp ? 1 : 0);

	for(unsigned sidx = 0; sidx < m; ++sidx) {
		//! \todo add dependence of delay on distance
		target_t target = targetNeuron(patch, x, y, pcount, distance, angle);
		targets.at(sidx) = target.first;
		weights.at(sidx) = -rweight();
		delays.at(sidx) = delay(target.second);
		//std::cout << neuronIndex(patch, x, y) << " -> " << target.first << " d=" << target.second << "\n";
	}

	net->addSynapses(neuronIndex(patch, x, y),
			targets, delays, weights, isPlastic);
}




nemo::Configuration
configure(bool stdp, bool logging=true)
{
	nemo::Configuration conf;

	if(logging) {
		conf.enableLogging();
	}
	//! \todo make network report STDP function
	if(stdp) {
		std::vector<float> pre(20);
		std::vector<float> post(20);
		for(unsigned i = 0; i < 20; ++i) {
			float dt = float(i + 1);
			pre.at(i) = 1.0 * expf(-dt / 20.0f);
			pre.at(i) = -0.8 * expf(-dt / 20.0f);
		}
		conf.setStdpFunction(pre, post, 10.0, -10.0);
	}

	return conf;
}



nemo::Network*
construct(unsigned pcount, unsigned m, bool stdp, double sigma, bool logging=true)
{
	nemo::Network* net = new nemo::Network();

	/* The network is a torus which consists of pcount rectangular patches,
	 * each with dimensions height * width. The size of each patch is the same
	 * as the partition size on the device. */
	const unsigned height = PATCH_HEIGHT;
	const unsigned width = PATCH_WIDTH;
	//! \todo check that this matches partition size
	
	rng_t rng;

	/* 80% of neurons are excitatory, 20% inhibitory. The spatial distribution
	 * of excitatory and inhibitory neurons is uniformly random. */
	boost::variate_generator<rng_t&, boost::bernoulli_distribution<double> >
		isExcitatory(rng, boost::bernoulli_distribution<double>(0.8));

	/* Postsynaptic neurons have a gaussian distribution of distance from
	 * presynaptic. Target neurons are in practice drawn from a 2D laplacian.
	 */ 

	/* Most inhibitory synapses are local. 95% fall within a patch. */
	double sigmaIn = width/2;
	grng_t distanceIn(rng, boost::normal_distribution<double>(0, sigmaIn));
	
	/* The user can control the distribution of the exitatory synapses */
	assert(sigma >= sigmaIn);
	grng_t distanceEx(rng, boost::normal_distribution<double>(0, sigma));

	urng_t angle(rng, boost::uniform_real<double>(0, 2*PI));

	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));

	unsigned exCount = 0;
	unsigned inCount = 0;

	for(unsigned p = 0; p < pcount; ++p) {
		if(logging) {
			std::cout << "Partition " << p << std::endl;
		}
		for(unsigned y = 0; y < height; ++y) {
			for(unsigned x = 0; x < width; ++x) {
				unsigned nidx = neuronIndex(p, x, y);
				if(isExcitatory()) {
					addExcitatoryNeuron(net, nidx, randomParameter);
					addExcitatorySynapses(net, p, x, y, pcount, m, stdp,
							distanceEx, angle, randomParameter);
					exCount++;
				} else {
					addInhibitoryNeuron(net, nidx, randomParameter);
					addInhibitorySynapses(net, p, x, y, pcount, m, false,
							distanceIn, angle, randomParameter);
					inCount++;
				}
			}
		}
	}

	if(logging) {
		std::cout << "Constructed network with " << exCount + inCount << " neurons\n"
			<< "\t" << exCount << " excitatory\n"		
			<< "\t" << inCount << " inhibitory\n";
		//! \todo report connectivity stats as well
	}

	return net;
}




void
simulate(nemo::Simulation* sim, unsigned pcount, unsigned m, bool stdp)
{
	const unsigned MS_PER_SECOND = 1000;

	//! \todo fix timing code in kernel so that we don't have to force data onto device
	sim->stepSimulation();
#ifdef INCLUDE_TIMING_API
	sim->resetTimer();
#endif

	/* Run for a few seconds to warm up the network */
	std::cout << "Running simulation (warming up)...";
	for(unsigned s=0; s < 5; ++s) {
		for(unsigned ms = 0; ms < MS_PER_SECOND; ++ms) {
			sim->stepSimulation();
		}
		sim->flushFiringBuffer();
	}
#ifdef INCLUDE_TIMING_API
	std::cout << "[" << sim->elapsedWallclock() << "ms elapsed]" << std::endl;
	sim->resetTimer();
#endif

	unsigned seconds = 10;

	/* Run once without reading data back, in order to estimate PCIe overhead */ 
	std::cout << "Running simulation (without reading data back)...";
	for(unsigned s=0; s < seconds; ++s) {
		std::cout << s << " ";
		for(unsigned ms = 0; ms < MS_PER_SECOND; ++ms) {
			sim->stepSimulation();
		}
		sim->flushFiringBuffer();
	}
#ifdef INCLUDE_TIMING_API
	long int elapsedTiming = sim->elapsedWallclock();
	sim->resetTimer();
	std::cout << "[" << elapsedTiming << "ms elapsed]" << std::endl;
#endif

	/* Dummy buffers for firing data */
	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* fired;

	std::cout << "Running simulation (gathering performance data)...";
	unsigned nfired = 0;
	for(unsigned s=0; s < seconds; ++s) {
		std::cout << s << " ";
		for(unsigned ms=0; ms<1000; ++ms) {
			sim->stepSimulation();
		}
		sim->readFiring(&cycles, &fired);
		nfired += fired->size();
	}
#ifdef INCLUDE_TIMING_API
	long int elapsedData = sim->elapsedWallclock();
	std::cout << "[" << elapsedData << "ms elapsed]" << std::endl;
#endif

	unsigned long narrivals = nfired * m;
	double f = (double(nfired) / (pcount * PATCH_SIZE)) / double(seconds);

#ifdef INCLUDE_TIMING_API
	/* Throughput is measured in terms of the number of spike arrivals per
	 * wall-clock second */
	unsigned long throughputNoPCI = MS_PER_SECOND * narrivals / elapsedTiming;
	unsigned long throughputPCI = MS_PER_SECOND * narrivals / elapsedData;

	double speedupNoPCI = double(seconds*MS_PER_SECOND)/elapsedTiming;
	double speedupPCI = double(seconds*MS_PER_SECOND)/elapsedData;
#endif

	std::cout << "Total firings: " << nfired << std::endl;
	std::cout << "Avg. firing rate: " << f << "Hz\n";
	std::cout << "Spike arrivals: " << narrivals << std::endl;
#ifdef INCLUDE_TIMING_API
	std::cout << "Performace both with and without PCI traffic overheads:\n";
	std::cout << "Approx. throughput: " << throughputPCI/1000000 << "/"
			<< throughputNoPCI/1000000 << "Ma/s (million spike arrivals per second)\n";
	std::cout << "Speedup wrt real-time: " << speedupPCI << "/"
			<< speedupNoPCI << std::endl;
#endif

	//sim->stopSimulation();
}



/* Simulate for some time, writing firing data to file */
void
simulateToFile(nemo::Simulation* sim, unsigned pcount, unsigned m, bool stdp, const char* firingFile)
{
	/* Dummy buffers for firing data */
	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* fired;

	std::cout << "Running simulation (gathering performance data)...";
	unsigned nfired = 0;
	for(unsigned ms=0; ms<1000; ++ms) {
		sim->stepSimulation();
	}
	sim->readFiring(&cycles, &fired);

	std::ofstream file;
	file.open(firingFile);
	for(size_t i = 0; i < cycles->size(); ++i) {
		file << cycles->at(i) << " " << fired->at(i) << "\n";
	}
	file.close();
}


#ifndef TORUS_NO_MAIN

int
main(int argc, char* argv[])
{
	if(argc != 3) {
		fprintf(stderr, "Usage: run pcount sigma\n");
		exit(-1);
	}

	unsigned pcount = atoi(argv[1]);
	unsigned sigma = atoi(argv[2]);
	assert(sigma >= PATCH_WIDTH/2);

	//! \todo get RNG seed option from command line
	//! \todo otherwise seed from system time

	fprintf(stderr, "%s w/%u partitions\n", argv[1], pcount);

	//! \todo add stdp command-line option
	bool stdp = false;
	unsigned m = 1000; // synapses per neuron
	
	nemo::Network* net = construct(pcount, m, stdp, sigma);
	nemo::Configuration conf = configure(stdp);
	nemo::Simulation* sim = nemo::Simulation::create(*net, conf);
	simulate(sim, pcount, m, stdp);
	//simulateToFile(net, pcount, m, stdp, "firing.dat");
	delete net;
}

#endif

