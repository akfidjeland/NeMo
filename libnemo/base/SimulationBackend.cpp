/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "SimulationBackend.hpp"
#include "Network.hpp"
#include "NetworkImpl.hpp"
#include "Configuration.hpp"
#include "exception.hpp"
#include "nemo_error.h"
#include "fixedpoint.hpp"

#include <CudaSimulation.hpp>

namespace nemo {

SimulationBackend*
SimulationBackend::create(const Network& net, const Configuration& conf)
{
	return create(*net.m_impl, conf);
}



/* Sometimes using the slightly lower-level interface provided by NetworkImpl
 * makes sense (see e.g. nemo::mpi::Worker), so provide an overload of 'create'
 * that takes such an object directly. */
SimulationBackend*
SimulationBackend::create(const NetworkImpl& net, const Configuration& conf)
{
	if(net.neuronCount() == 0) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot create simulation from empty network");
		return NULL;
	}
	int dev = cuda::Simulation::selectDevice();
	if(dev == -1) {
		throw nemo::exception(NEMO_CUDA_ERROR, "Failed to create simulation");
	}
	return new cuda::Simulation(net, *conf.m_impl);
}



SimulationBackend::~SimulationBackend()
{
	;
}


void
SimulationBackend::step(const std::vector<unsigned>& fstim, const std::vector<float>& istim)
{
	setFiringStimulus(fstim);
	setCurrentStimulus(istim);
	step();
}



void
SimulationBackend::setCurrentStimulus(const std::vector<float>& current)
{
	unsigned fbits = getFractionalBits();
	size_t len = current.size();
	/*! \todo allocate this only once */
	std::vector<fix_t> fx_current(len);
	for(size_t i = 0; i < len; ++i) {
		fx_current.at(i) = fx_toFix(current.at(i), fbits);
	}
	setCurrentStimulus(fx_current);
}




}
