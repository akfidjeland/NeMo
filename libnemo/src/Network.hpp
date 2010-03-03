#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include "except.hpp"
#include "DeviceAssertions.hpp"

namespace nemo {

class Network
{
	public :

		/*! \return
		 * 		new Network object, or NULL if no suitable CUDA device was
		 * 		found. For CUDA device 0 will be used.  */
		static Network* create(bool setReverse, unsigned maxReadPeriod);

		virtual ~Network() {};

		/*
		 * CONFIGURATION
		 */

		/*! Switch on logging and send output to stdout */
		virtual void logToStdout() = 0;

		virtual void enableStdp(
				std::vector<float> prefire,
				std::vector<float> postfire,
				float minWeight, float maxWeight) = 0;

		/*
		 * NETWORK CONSTRUCTION
		 */

		virtual void addNeuron(unsigned int idx,
				float a, float b, float c, float d,
				float u, float v, float sigma) = 0;

		virtual void addSynapses(
				unsigned source,
				const std::vector<unsigned>& targets,
				const std::vector<unsigned>& delays,
				const std::vector<float>& weights,
				//! \todo change to bool
				const std::vector<unsigned char> isPlastic) = 0;

		/*
		 * NETWORK SIMULATION
		 */

		virtual void startSimulation() = 0;

		virtual void stepSimulation(const std::vector<unsigned>& fstim)
			throw(class DeviceAssertionFailure, std::logic_error) = 0;

		virtual void applyStdp(float reward) = 0;

		/*! Read all firing data buffered on the device since the previous
		 * call to this function (or the start of simulation if this is the
		 * first call). The return vectors are valid until the next call to
		 * this function.
		 *
		 * \return
		 * 		Total number of cycles for which we return firing. The caller
		 * 		would most likely already know what this should be, so can use
		 * 		this for sanity checking.
		 */
		virtual unsigned readFiring(
				const std::vector<unsigned>** cycles,
				const std::vector<unsigned>** nidx) = 0;

		virtual void flushFiringBuffer() = 0;

		/*
		 * TIMING
		 */

		/*! \return number of milliseconds of wall-clock time elapsed since
		 * first simulation step (or last timer reset). */
		virtual unsigned long elapsedWallclock() const = 0;

		/*! \return number of milliseconds of simulated time elapsed since first
		 * simulation step (or last timer reset) */
		virtual unsigned long elapsedSimulation() const = 0;

		virtual void resetTimer() = 0;
};

};

#endif
