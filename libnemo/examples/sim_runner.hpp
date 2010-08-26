#ifndef SIM_RUNNER_HPP
#define SIM_RUNNER_HPP

#include <ostream>
#include <boost/program_options.hpp>
#include <nemo.hpp>

/*! Run simulation and report performance results
 *
 * The run does a bit of warming up and does measures performance for some time
 * without reading data back.
 *
 * \param n
 * 		Number of neurons in the network
 * \param m
 * 		Number of synapses per neuron. This must be the same for each neuron in
 * 		order for the throughput-measurement to work out. This constraint is
 * 		not checked.
 * \param stdp
 * 		Period (in ms) between STDP applications. If 0, run without STDP.
 */
void benchmark(nemo::Simulation* sim, unsigned n, unsigned m, unsigned stdp=0);


/*! Run simulation for some time, writing data to output stream
 *
 * \param time_ms
 * 		Number of milliseconds simulation should be run
 * \param stdp
 * 		Period (in ms) between STDP applications. If 0, run without STDP.
 */
void
simulate(nemo::Simulation* sim, unsigned time_ms, unsigned stdp=0, std::ostream& out=std::cout);


/*! Run simulation for some time, writing data to output to file. Existing file
 * contents will be overwritten.
 *
 * \see simulate
 */
void
simulateToFile(nemo::Simulation* sim, unsigned time_ms, unsigned stdp, const char* firingFile);


/*! \return a 'standard' configuration using the default backend and STDP
 * optionally enabled. */
nemo::Configuration configuration(bool stdp);


/*! \return a 'standard' configuration with the specified backend and STDP
 * optionally enabled. */
nemo::Configuration configuration(bool stdp, backend_t backend);


/* Return common program options */
boost::program_options::options_description
commonOptions();

typedef boost::program_options::variables_map vmap;

vmap
processOptions(int argc, char* argv[], const boost::program_options::options_description& desc);




#endif
