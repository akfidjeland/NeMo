#include <iostream>
#include <fstream>

#include <nemo.hpp>


void
simulate(nemo::Simulation* sim, unsigned n, unsigned m)
{
	const unsigned MS_PER_SECOND = 1000;

#ifdef INCLUDE_TIMING_API
	sim->resetTimer();
#endif

	/* Run for a few seconds to warm up the network */
	std::cout << "Running simulation (warming up)...";
	for(unsigned s=0; s < 5; ++s) {
		for(unsigned ms = 0; ms < MS_PER_SECOND; ++ms) {
			sim->step();
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
			sim->step();
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
	unsigned long nfired = 0;
	for(unsigned s=0; s < seconds; ++s) {
		std::cout << s << " ";
		for(unsigned ms=0; ms<1000; ++ms) {
			sim->step();
		}
		sim->readFiring(&cycles, &fired);
		nfired += fired->size();
	}
#ifdef INCLUDE_TIMING_API
	long int elapsedData = sim->elapsedWallclock();
	std::cout << "[" << elapsedData << "ms elapsed]" << std::endl;
#endif

	unsigned long narrivals = nfired * m;
	double f = (double(nfired) / n) / double(seconds);

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
}



void
simulateToFile(nemo::Simulation* sim, unsigned time_ms, const char* firingFile)
{
	/* Dummy buffers for firing data */
	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* fired;

	unsigned nfired = 0;
	for(unsigned ms=0; ms<time_ms; ++ms) {
		sim->step();
	}
	sim->readFiring(&cycles, &fired);

	std::ofstream file;
	file.open(firingFile);
	for(size_t i = 0; i < cycles->size(); ++i) {
		file << cycles->at(i) << " " << fired->at(i) << "\n";
	}
	file.close();
}

