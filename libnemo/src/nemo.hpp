#ifndef NEMO_HPP
#define NEMO_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file nemo.hpp

#include <vector>

namespace nemo {


/*! \class Network
 * \brief C++ API for nemo
 *
 * To use, create an network object using \ref nemo::Network::create, configure
 * it using any of the configuration commands, construct a network by adding
 * neurons and synapses, and finally run the simulation.
 *
 * \section errors Error Handling
 *
 * Internal errors are signaled by exceptions. Thrown exceptions are all
 * subclasses of std::exception.
 */
class Network
{
	public :

		/*! \name Initialisation */
		/* \{ */ // begin init section

		/*!
		 * \param usingStdp
		 * 		Will STDP be used?
		 *
		 * \return
		 * 		new Network object, or NULL if no suitable CUDA device was
		 * 		found. For CUDA device 0 will be used.  */
		static Network* create(bool setReverse);

		/* \} */ // end init section

		virtual ~Network();

		/*! \name Configuration */
		/* \{ */ // begin configuration

		/*! Switch on logging and send output to stdout */
		virtual void logToStdout() = 0;

		virtual void enableStdp(
				std::vector<float> prefire,
				std::vector<float> postfire,
				float minWeight, float maxWeight) = 0;

		/*! Set the size of the firing buffer such that it can contain a fixed
		 * number of \a cycles worth of firing data before overflowing. */
		virtual void setFiringBufferLength(unsigned cycles) = 0;

		/*! \return the number of cycles the firing buffer can hold */
		virtual unsigned getFiringBufferLength() const = 0;

		/* \} */ // end configuration

		/*! \name Construction
		 *
		 * Networks are constructed by adding individual neurons, and single or
		 * groups of synapses to the network. Neurons are given indices (from
		 * 0) which should be unique for each neuron. When adding synapses the
		 * source or target neurons need not necessarily exist yet, but should
		 * be defined before the network is finalised.
		 * \{ */

		/*! Add a single neuron to the network
		 *
		 * The neuron uses the Izhikevich neuron model. See E. M. Izhikevich
		 * "Simple model of spiking neurons", \e IEEE \e Trans. \e Neural \e
		 * Networks, vol 14, pp 1569-1572, 2003 for a full description of the
		 * model and the parameters.
		 *
		 * \param idx
		 * 		Neuron index. This should be unique
		 * \param a
		 * 		Time scale of the recovery variable \a u
		 * \param b
		 * 		Sensitivity to sub-threshold fluctutations in the membrane
		 * 		potential \a v
		 * \param c
		 * 		After-spike reset value of the membrane potential \a v
		 * \param d
		 * 		After-spike reset of the recovery variable \a u
		 * \param u
		 * 		Initial value for the membrane recovery variable
		 * \param v
		 * 		Initial value for the membrane potential
		 * \param sigma
		 * 		Parameter for a random gaussian per-neuron process which
		 * 		generates random input current drawn from an N(0,\a sigma)
		 * 		distribution. If set to zero no random input current will be
		 * 		generated.
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
		 * 		Specifies for each synapse whether or not it is plastic.
		 * 		See section on STDP.
		 *
		 * \pre
		 * 		\a targets, \a delays, \a weights, and \a isPlastic have the
		 * 		same length
		 */
		virtual void addSynapses(
				unsigned source,
				const std::vector<unsigned>& targets,
				const std::vector<unsigned>& delays,
				const std::vector<float>& weights,
				const std::vector<unsigned char> isPlastic) = 0;
		//! \todo change to bool

		/* \} */ // end construction group

		/*! \name Simulation
		 * \{ */

		/*! Finalise network construction to prepare it for
		 * simulation
		 *
		 * After specifying the network in the construction stage, it may need
		 * to be set up on the backend, and optimised etc. This can be
		 * time-consuming if the network is large. Calling the simulation
		 * initialization function causes all this setup to be done. Calling
		 * this function is not strictly required as the setup will be done the
		 * first time any simulation function is called. */
		virtual void initSimulation() = 0;

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

		/*! \name Simulation (firing)
		 *
		 * The indices of the fired neurons are buffered on the device, and can
		 * be read back at run-time. The desired size of the buffer is
		 * specified when constructing the network. Each read empties the
		 * buffer. To avoid overflow if the firing data is not needed, call
		 * \ref flushFiringBuffer periodically.
		 *
		 * \{ */

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
		 *		Total number of cycles for which we return firing. The caller
		 *		would most likely already know what this should be, so can use
		 *		this for sanity checking.
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

		/*! \name Simulation (queries)
		 *
		 * If STDP is enabled, the synaptic weights may change
		 * at run-time. The user can read these back on a
		 * per-(source) neuron basis.
		 *
		 * \{ */

		/*! Return synapse data for a specific source neuron. If STDP is
		 * enabled the weights may change at run-time. The order of synapses in
		 * each returned vector is guaranteed to be the same on subsequent
		 * calls to this function. The output vectors are
		 * valid until the next call to this function.
		 *
		 * \post The output vectors all have the same length */
		virtual void getSynapses(unsigned sourceNeuron,
				const std::vector<unsigned>** targetNeuron,
				const std::vector<unsigned>** delays,
				const std::vector<float>** weights,
				const std::vector<unsigned char>** plastic) = 0;

		/* \} */ // end simulation (queries) section

		/*! \name Simulation (timing)
		 *
		 * The simulation has two internal timers which keep track of the
		 * elapsed \e simulated time and \e wallclock time. Both timers measure
		 * from the first simulation step, or from the last timer reset,
		 * whichever comes last.
		 *
		 * \{ */

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
