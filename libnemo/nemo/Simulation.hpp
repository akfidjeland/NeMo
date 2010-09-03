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
#include <nemo/config.h>

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
class NEMO_BASE_DLL_PUBLIC Simulation
{
	public :

		virtual ~Simulation();

		/*! Run simulation for a single cycle (1ms)
		 *
		 * \param fstim
		 * 		An optional list of neurons, which will be forced to fire this
		 * 		cycle.
		 * \param istim
		 * 		Optional per-neuron vector specifying externally provided input
		 * 		current for this cycle.
		 * \return
		 * 		List of neurons which fired this cycle. The referenced data is
		 * 		valid until the next call to step.
		 */
		virtual const std::vector<unsigned>& step(
				const std::vector<unsigned>& fstim = std::vector<unsigned>(),
				const std::vector<float>& istim = std::vector<float>()) = 0;

		const std::vector<unsigned>& step2() { return step(); }

		/*! Update synapse weights using the accumulated STDP statistics
		 *
		 * \param reward
		 * 		Multiplier for the accumulated weight change
		 */
		virtual void applyStdp(float reward) = 0;

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
		 * calls to this function. The output vectors are valid until the next
		 * call to this function.
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
