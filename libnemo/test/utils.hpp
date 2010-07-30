#ifndef NEMO_TEST_UTILS_HPP
#define NEMO_TEST_UTILS_HPP

/* Run simulation for given length and return result in output vector */
void
runSimulation(
		const nemo::Network* net,
		nemo::Configuration conf,
		unsigned seconds,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx);

void
compareSimulationResults(
		const std::vector<unsigned>& cycles1,
		const std::vector<unsigned>& nidx1,
		const std::vector<unsigned>& cycles2,
		const std::vector<unsigned>& nidx2);

#endif
