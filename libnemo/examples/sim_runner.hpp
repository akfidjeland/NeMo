#ifndef SIM_RUNNER_HPP
#define SIM_RUNNER_HPP

/*! Run simulation and report performance results
 *
 * \note STDP is not applied, but if it was enabled when configuring the
 * simulation, the STDP statistics will be gathered on the device.
 *
 * \param n
 * 		Number of neurons in the network
 * \param m
 * 		Number of synapses per neuron. This must be the same for each neuron in
 * 		order for the throughput-measurement to work out. This constraint is
 * 		not checked.
 */
void simulate(nemo::Simulation* sim, unsigned n, unsigned m);


/*! Run simulation for some time, writing firing data to file
 *
 * \note STDP is not applied, but if it was enabled when configuring the
 * simulation, the STDP statistics will be gathered on the device.
 *
 * \param time_ms
 * 		Number of milliseconds simulation should be run
 * \param firingFile
 * 		File to which firing data is written. Existing contents will be
 * 		overwritten.
 */
void
simulateToFile(nemo::Simulation* sim, unsigned time_ms, const char* firingFile);

#endif
