#ifndef NEMO_SIMULATION_BACKEND_HPP
#define NEMO_SIMULATION_BACKEND_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file SimulationBackend.hpp

#include "Simulation.hpp"
#include "internal_types.h"

namespace nemo {

class Network;
class Configuration;


/*! \class SimulationBackend
 *
 * This is the lower-level interface that backends should implement. This
 * interface is not exposed in the public API, which instead uses the
 * Simulation base class interface.
 */
class NEMO_BASE_DLL_PUBLIC SimulationBackend : public Simulation
{
	public :

		virtual ~SimulationBackend();

		/*! \copydoc nemo::Simulation::getFiringBufferLength */
		virtual unsigned getFiringBufferLength() const = 0;

		virtual unsigned getFractionalBits() const = 0;

		/*! Set firing stimulus for the next simulation step.
		 *
		 * The behaviour is undefined if this function is called multiple times
		 * between two calls to \a step */
		virtual void setFiringStimulus(const std::vector<unsigned>& nidx) = 0;

		/*! Set per-neuron input current on the device and set the relevant
		 * member variable containing the device pointer. If there is no input
		 * the device pointer is NULL.
		 *
		 * The behaviour is undefined if this function is called multiple times
		 * between two calls to \a step */
		void setCurrentStimulus(const std::vector<float>& current);

		/*! Set per-neuron input current on the device and set the relevant
		 * member variable containing the device pointer. If there is no input
		 * the device pointer is NULL.
		 *
		 * This function should only be called once per cycle.
		 *
		 * Pre: the input vector uses the same fixed-point format as the backend */
		virtual void setCurrentStimulus(const std::vector<fix_t>& current) = 0;

		/*! Perform a single simulation step, using any stimuli (firing
		 * and current) provided by the caller after the previous call
		 * to step */
		virtual void step() = 0;

		/*! \copydoc nemo::Simulation::step */
		virtual void step(
				const std::vector<unsigned>& fstim,
				const std::vector<float>& istim);

		/*! \copydoc nemo::Simulation::applyStdp */
		virtual void applyStdp(float reward) = 0;

		/*! \copydoc nemo::Simulation::readFiring */
		virtual unsigned readFiring(
				const std::vector<unsigned>** cycles,
				const std::vector<unsigned>** nidx) = 0;

		/*! \copydoc nemo::Simulation::flushFiringBuffer */
		virtual void flushFiringBuffer() = 0;

		/*! \copydoc nemo::Simulation::getSynapses */
		virtual void getSynapses(unsigned sourceNeuron,
				const std::vector<unsigned>** targetNeuron,
				const std::vector<unsigned>** delays,
				const std::vector<float>** weights,
				const std::vector<unsigned char>** plastic) = 0;

		/*! \copydoc nemo::Simulation::elapsedWallclock */
		virtual unsigned long elapsedWallclock() const = 0;

		/*! \copydoc nemo::Simulation::elapsedSimulation */
		virtual unsigned long elapsedSimulation() const = 0;

		/*! \copydoc nemo::Simulation::resetTimer */
		virtual void resetTimer() = 0;

	protected :

		SimulationBackend() { };

	private :

		/* Disallow copying of SimulationBackend object */
		SimulationBackend(const Simulation&);
		SimulationBackend& operator=(const Simulation&);

};

};

#endif
