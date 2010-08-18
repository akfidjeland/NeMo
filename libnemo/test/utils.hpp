#ifndef NEMO_TEST_UTILS_HPP
#define NEMO_TEST_UTILS_HPP

#include <nemo.hpp>

/* Run simulation for given length and return result in output vector */
void
runSimulation(
		const nemo::Network* net,
		nemo::Configuration conf,
		unsigned seconds,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx,
		bool stdp,
		std::vector<unsigned> initFiring = std::vector<unsigned>());

void
compareSimulationResults(
		const std::vector<unsigned>& cycles1,
		const std::vector<unsigned>& nidx1,
		const std::vector<unsigned>& cycles2,
		const std::vector<unsigned>& nidx2);


void
setBackend(backend_t, nemo::Configuration& conf);

nemo::Configuration
configuration(bool stdp, unsigned partitionSize, backend_t backend = NEMO_BACKEND_CUDA);

/* Add a 'standard' excitatory neuron with fixed parameters */
void
addExcitatoryNeuron(unsigned nidx, nemo::Network& net, float sigma=0.0f);

#endif
