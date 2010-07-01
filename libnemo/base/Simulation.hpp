#ifndef NEMO_SIMULATION_HPP
#define NEMO_SIMULATION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file Simulation.hpp

#include <vector>
#include <nemo_config.h>

namespace nemo {

class Network;
class Configuration;


/*! \class Simulation
 *
 * To use, create a network object using \ref nemo::Simulation::create, configure
 * it using any of the configuration commands, construct a network by adding
 * neurons and synapses, and finally run the simulation.
 *
 * Internal errors are signaled by exceptions. Thrown exceptions are all
 * subclasses of std::exception.
 *
 * \ingroup cpp-api
 */
class DLL_PUBLIC Simulation
{
	public :

		/*!
		 * \return
		 * 		new Simulation object, or NULL if no suitable CUDA device was
		 * 		found. */
		static Simulation* create(const Network& net, const Configuration&);
		static Simulation* create(const class NetworkImpl& net, const Configuration& conf);

		virtual ~Simulation() = 0;

		/*! \return the number of cycles the firing buffer can hold */
		virtual unsigned getFiringBufferLength() const = 0;

		/*! Run simulation for a single cycle (1ms)
		 *
		 * \param fstim
		 * 		An optional list of neurons, which will be forced to fire this
		 * 		cycle.
		 * \param istim
		 * 		Optional per-neuron vector specifying externally provided input
		 * 		current for this cycle.
		 */
		virtual void step(
				// fstim is optional due to low-level overloaded step()
				const std::vector<unsigned>& fstim,
				const std::vector<float>& istim = std::vector<float>());

		/* Low-level simulation interface */
		virtual void setFiringStimulus(const std::vector<unsigned>& nidx) = 0;
		virtual void setCurrentStimulus(const std::vector<float>& current) = 0;
		virtual void step() = 0;

		/*! Update synapse weights using the accumulated STDP statistics
		 *
		 * \param reward
		 * 		Multiplier for the accumulated weight change
		 */
		virtual void applyStdp(float reward) = 0;

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

	protected :

		Simulation() { };

	private :

		/* Disallow copying of Simulation object */
		Simulation(const Simulation&);
		Simulation& operator=(const Simulation&);

};

};

#endif
