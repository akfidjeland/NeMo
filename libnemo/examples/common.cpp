#include <iostream>
#include <fstream>
#include <cmath>

#include "common.hpp"


void
benchmark(nemo::Simulation* sim, unsigned n, unsigned m, unsigned stdp)
{
	const unsigned MS_PER_SECOND = 1000;

#ifdef NEMO_TIMING_ENABLED
	sim->resetTimer();
#endif

	unsigned t = 0;

	/* Run for a few seconds to warm up the network */
	std::cout << "Running simulation (warming up)...";
	for(unsigned s=0; s < 5; ++s) {
		for(unsigned ms = 0; ms < MS_PER_SECOND; ++ms, ++t) {
			sim->step();
		}
		if(stdp && t % stdp == 0) {
			sim->applyStdp(1.0);
		}
	}
#ifdef NEMO_TIMING_ENABLED
	std::cout << "[" << sim->elapsedWallclock() << "ms elapsed]";
	sim->resetTimer();
#endif
	std::cout << std::endl;

	unsigned seconds = 10;

	/* Run once without reading data back, in order to estimate PCIe overhead */ 
	std::cout << "Running simulation (without reading data back)...";
	for(unsigned s=0; s < seconds; ++s) {
		std::cout << s << " ";
		for(unsigned ms = 0; ms < MS_PER_SECOND; ++ms, ++t) {
			sim->step();
		}
		if(stdp && t % stdp == 0) {
			sim->applyStdp(1.0);
		}
	}
#ifdef NEMO_TIMING_ENABLED
	long int elapsedTiming = sim->elapsedWallclock();
	sim->resetTimer();
	std::cout << "[" << elapsedTiming << "ms elapsed]";
#endif
	std::cout << std::endl;

	std::cout << "Running simulation (gathering performance data)...";
	unsigned long nfired = 0;
	for(unsigned s=0; s < seconds; ++s) {
		std::cout << s << " ";
		for(unsigned ms=0; ms<1000; ++ms, ++t) {
			const std::vector<unsigned>& fired = sim->step();
			nfired += fired.size();
		}
		if(stdp && t % stdp == 0) {
			sim->applyStdp(1.0);
		}
	}
#ifdef NEMO_TIMING_ENABLED
	long int elapsedData = sim->elapsedWallclock();
	std::cout << "[" << elapsedData << "ms elapsed]";
#endif
	std::cout << std::endl;

	unsigned long narrivals = nfired * m;
	double f = (double(nfired) / n) / double(seconds);

#ifdef NEMO_TIMING_ENABLED
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
#ifdef NEMO_TIMING_ENABLED
	std::cout << "Performace both with and without PCI traffic overheads:\n";
	std::cout << "Approx. throughput: " << throughputPCI/1000000 << "/"
			<< throughputNoPCI/1000000 << "Ma/s (million spike arrivals per second)\n";
	std::cout << "Speedup wrt real-time: " << speedupPCI << "/"
			<< speedupNoPCI << std::endl;
#endif
}



void
simulate(nemo::Simulation* sim, unsigned time_ms, unsigned stdp, std::ostream& out)
{
	for(unsigned ms=0; ms<time_ms; ) {
		const std::vector<unsigned>& fired = sim->step();
		for(std::vector<unsigned>::const_iterator fi = fired.begin(); fi != fired.end(); ++fi) {
			out << ms << " " << *fi << "\n";
		}
		ms += 1;
		if(stdp != 0 && ms % stdp == 0) {
			sim->applyStdp(1.0);
		}
	}
}



void
simulateToFile(nemo::Simulation* sim, unsigned time_ms, unsigned stdp, const char* firingFile)
{
	std::ofstream file;
	file.open(firingFile);
	simulate(sim, time_ms, stdp, file);
	file.close();
}



nemo::Configuration
configuration(bool stdp, bool log)
{
	nemo::Configuration conf;

	if(log) {
		conf.enableLogging();
	}

	if(stdp) {
		std::vector<float> pre(20);
		std::vector<float> post(20);
		for(unsigned i = 0; i < 20; ++i) {
			float dt = float(i + 1);
			pre.at(i) = 0.1 * expf(-dt / 20.0f);
			post.at(i) = -0.08 * expf(-dt / 20.0f);
		}
		conf.setStdpFunction(pre, post, -1.0, 1.0);
	}

	return conf;
}



nemo::Configuration
configuration(bool stdp, backend_t backend, bool log)
{
	nemo::Configuration conf = configuration(stdp, log);
	switch(backend) {
		case NEMO_BACKEND_CPU: conf.setCpuBackend(); break;
		case NEMO_BACKEND_CUDA: conf.setCudaBackend(); break;
		default:
			std::cerr << "Invalid backend specified\n";
			exit(-1);
	}
	return conf;
}



boost::program_options::options_description
commonOptions()
{
	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "print this message")
		("duration,t", po::value<unsigned>()->default_value(1000), "duration of simulation (ms)")
		("stdp", po::value<unsigned>()->default_value(0), "STDP application period (ms). If 0 do not use STDP")
		("verbose,v", po::value<unsigned>()->default_value(0), "Set verbosity level")
		("output-file,o", po::value<std::string>(), "output file for firing data")
		("list-devices", "print the available simulation devices")
		("benchmark", "report performance results instead of returning firing")
	;

	return desc;
}



void
listCudaDevices()
{
	unsigned dcount  = nemo::cudaDeviceCount();

	if(dcount == 0) {
		std::cout << "No CUDA devices available\n";
		return;
	}

	for(unsigned d = 0; d < dcount; ++d) {
		std::cout << d << ": " << nemo::cudaDeviceDescription(d) << std::endl;
	}
}




boost::program_options::variables_map
processOptions(int argc, char* argv[],
		const boost::program_options::options_description& desc)
{
	namespace po = boost::program_options;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")) {
		std::cout << "Usage:\n\t" << argv[0] << " [OPTIONS]\n\n";
		std::cout << desc << std::endl;
		exit(1);
	}

	if(vm.count("list-devices")) {
		listCudaDevices();
		exit(0);
	}

	return vm;
}
