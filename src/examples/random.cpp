/* Simple network with 1000 neurons with all-to-all connections with random
 * weights.

 * Author: Andreas K. Fidjeland <andreas.fidjeland@imperial.ac.uk>
 * Date: April 2010
 */

#include <vector>

#ifdef USING_MAIN
#	include <string>
#	include <iostream>
#	include <fstream>
#	include <boost/program_options.hpp>
#	include <boost/scoped_ptr.hpp>
#	include <examples/common.hpp>
#endif

#include <boost/random.hpp>
#include <nemo.hpp>

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

namespace nemo {
	namespace random {

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
	float b = 0.25f - 0.05f * r2;
	float c = v;
	float d = 2.0f;
	float u = b * v;
	float sigma = 2.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
}



nemo::Network*
construct(unsigned ncount, unsigned scount, unsigned dmax, bool stdp)
{
	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomTarget(rng, boost::uniform_int<>(0, ncount-1));
	uirng_t randomDelay(rng, boost::uniform_int<>(1, dmax));

	nemo::Network* net = new nemo::Network();

	for(unsigned nidx=0; nidx < ncount; ++nidx) {
		if(nidx < (ncount * 4) / 5) { // excitatory
			addExcitatoryNeuron(net, nidx, randomParameter);
			for(unsigned s = 0; s < scount; ++s) {
				net->addSynapse(nidx, randomTarget(), randomDelay(), 0.5f * float(randomParameter()), stdp);
			}
		} else { // inhibitory
			addInhibitoryNeuron(net, nidx, randomParameter);
			for(unsigned s = 0; s < scount; ++s) {
				net->addSynapse(nidx, randomTarget(), 1U, float(-randomParameter()), 0);
			}
		}
	}
	return net;
}

	} // namespace random
} // namespace nemo


#ifdef USING_MAIN


#define LOG(cond, ...) if(cond) { fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); }


int
main(int argc, char* argv[])
{
	namespace po = boost::program_options;

	try {

		po::options_description desc = commonOptions();
		desc.add_options()
			("neurons,n", po::value<unsigned>()->default_value(1000), "number of neurons")
			("synapses,m", po::value<unsigned>()->default_value(1000), "number of synapses per neuron")
			("dmax,d", po::value<unsigned>()->default_value(1), "maximum excitatory delay,  where delays are uniform in range [1, dmax]")
		;

		po::variables_map vm = processOptions(argc, argv, desc);

		unsigned ncount = vm["neurons"].as<unsigned>();
		unsigned scount = vm["synapses"].as<unsigned>();
		unsigned dmax = vm["dmax"].as<unsigned>();
		unsigned duration = vm["duration"].as<unsigned>();
		unsigned stdp = vm["stdp-period"].as<unsigned>();
		unsigned verbose = vm["verbose"].as<unsigned>();
		bool runBenchmark = vm.count("benchmark") != 0;

		std::ofstream file;
		std::string filename;

		if(vm.count("output-file")) {
			filename = vm["output-file"].as<std::string>();
			file.open(filename.c_str()); // closes on destructor
		}

		std::ostream& out = filename.empty() ? std::cout : file;

		LOG(verbose, "Constructing network");
		boost::scoped_ptr<nemo::Network> net(nemo::random::construct(ncount, scount, dmax, stdp != 0));
		LOG(verbose, "Creating configuration");
		nemo::Configuration conf = configuration(vm);
		LOG(verbose, "Simulation will run on %s", conf.backendDescription());
		LOG(verbose, "Creating simulation");
		boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
		LOG(verbose, "Running simulation");
		if(runBenchmark) {
			benchmark(sim.get(), ncount, scount, vm);
		} else {
			simulate(sim.get(), duration, stdp, out);
		}
		LOG(verbose, "Simulation complete");
		return 0;
	} catch(std::exception& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	} catch(...) {
		std::cerr << "random: An unknown error occurred\n";
		return -1;
	}

}

#endif
