#ifndef NETWORK_HPP
#define NETWORK_HPP

//! \file Network.hpp

#include <vector>

namespace nemo {

/*! \brief Network class */
class Network
{
	public :

		/*! \name Initialisation */
		/* \{ */ // begin init section

		/*! \return
		 * 		new Network object, or NULL if no suitable CUDA device was
		 * 		found. For CUDA device 0 will be used.  */
		static Network* create(bool setReverse, unsigned maxReadPeriod);

		/* \} */ // end init section

		virtual ~Network();

		/*
		 * CONFIGURATION
		 */

		/*! \name Configuration */
		/* \{ */ // begin configuration

		/*! Switch on logging and send output to stdout */
		virtual void logToStdout() = 0;

		virtual void enableStdp(
				std::vector<float> prefire,
				std::vector<float> postfire,
				float minWeight, float maxWeight) = 0;

		/* \} */ // end configuration

		/*!\name Construction
		 *
		 * Networks are constructed by adding individual
		 * neurons, and single or groups of synapses to the
		 * network. Neurons are given indices (from 0) which
		 * should be unique for each neuron. When adding
		 * synapses the source or target neurons need not
		 * necessarily exist yet, but should be defined before
		 * the network is finalised.
		 * \{ */

		/*! Add a single neuron to the network
		 *
		 * The neuron uses the Izhikevich neuron model. See E. M. Izhikevich "Simple
		 * model of spiking neurons", \e IEEE \e Trans. \e Neural \e Networks, vol 14,
		 * pp 1569-1572, 2003 for a full description of the model and the parameters.
		 *
		 * \param a
		 * 		Time scale of the recovery variable \a u
		 * \param b
		 * 		Sensitivity to sub-threshold fluctutations in the membrane potential \a v
		 * \param c
		 * 		After-spike reset value of the membrane potential \a v
		 * \param d
		 * 		After-spike reset of the recovery variable \a u
		 * \param u
		 * 		Initial value for the membrane recovery variable
		 * \param v
		 * 		Initial value for the membrane potential
		 * \param sigma
		 * 		Parameter for a random gaussian per-neuron process which generates
		 * 		random input current drawn from an N(0,\a sigma) distribution. If set
		 * 		to zero no random input current will be generated.
		 */
		virtual void addNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma) = 0;

		/*! Add to the network a group of synapses with the same presynaptic neuron
		 *
		 * \param source
		 * 		Index of source neuron
		 * \param targets
		 * 		Indices of target neurons
		 * \param delays
		 * 		Synapse conductance delays in milliseconds
		 * \param weights
		 * 		Synapse weights
		 * \param isPlastic
		 * 		Specifies for each synapse whether or not it is plastic. See section on STDP.
		 *
		 * \pre
		 * 		\a targets, \a delays, \a weights, and
		 * 		\a isPlastic have the same length
		 */
		virtual void addSynapses(
				unsigned source,
				const std::vector<unsigned>& targets,
				const std::vector<unsigned>& delays,
				const std::vector<float>& weights,
				//! \todo change to bool
				const std::vector<unsigned char> isPlastic) = 0;

		/* \} */ // end construction group

		/*
		 * NETWORK SIMULATION
		 */

		/*! \name Simulation \{ */

		/*! Finalise network construction to prepare it for
		 * simulation
		 *
		 * After specifying the network in the construction
		 * stage, it may need to be set up on the backend, and
		 * optimised etc. This can be time-consuming if the
		 * network is large. Calling \a nemo_start_simulation
		 * causes all this setup to be done. Calling this
		 * function is not strictly required as the setup will
		 * be done the first time any simulation function is
		 * called. */
		virtual void startSimulation() = 0;

		/*! Run simulation for a single cycle (1ms)
		 *
		 * \param fstim
		 * 		An optional list of neurons, which will be forced to fire this
		 * 		cycle.
		 */
		virtual void stepSimulation(const std::vector<unsigned>& fstim = std::vector<unsigned>()) = 0;

		/*! Update synapse weights using the accumulated STDP statistics
		 *
		 * \param reward
		 * 		Multiplier for the accumulated weight change
		 */
		virtual void applyStdp(float reward) = 0;

		/* \} */ // end simulation group

		/*! \name Simulation (firing) \{ */

		//! \todo return pairs instead here
		/*! Read all firing data buffered on the device since the previous
		 * call to this function (or the start of simulation if this is the
		 * first call). The return vectors are valid until the next call to
		 * this function.
		 *
		 * \param cycles The cycle numbers during which firing occured
		 * \param nidx The corresponding neuron indices
		 *
		 * \return
		 * 		Total number of cycles for which we return firing. The caller
		 * 		would most likely already know what this should be, so can use
		 * 		this for sanity checking.
		 */
		virtual unsigned readFiring(
				const std::vector<unsigned>** cycles,
				const std::vector<unsigned>** nidx) = 0;

		/*! If the user is not reading back firing, the firing output buffers
		 * should be flushed to avoid buffer overflow. The overflow is not
		 * harmful in that no memory accesses take place outside the buffer,
		 * but an overflow may result in later calls to readFiring returning
		 * non-sensical results. */
		virtual void flushFiringBuffer() = 0;

		/* \} */ // end simulation (firing)

		/*! \name Simulation (timing) \{ */

		/*! \return number of milliseconds of wall-clock time elapsed since
		 * first simulation step (or last timer reset). */
		virtual unsigned long elapsedWallclock() const = 0;

		/*! \return number of milliseconds of simulated time elapsed since first
		 * simulation step (or last timer reset) */
		virtual unsigned long elapsedSimulation() const = 0;

		/*! Reset both wall-clock and simulation timer */
		virtual void resetTimer() = 0;

		/* \} */ // end simulation (timing) section

};

};

#endif
