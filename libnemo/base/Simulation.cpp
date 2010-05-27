/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Simulation.hpp"
#include "Network.hpp"
#include "Configuration.hpp"
#include "exception.hpp"
#include "nemo_error.h"
#include <CudaSimulation.hpp>

namespace nemo {

Simulation*
Simulation::create(const Network& net, const Configuration& conf)
{
	if(net.neuronCount() == 0) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot create simulation from empty network");
		return NULL;
	}
	int dev = cuda::Simulation::selectDevice();
	return dev == -1 ? NULL : new cuda::Simulation(*net.m_impl, *conf.m_impl);
}


Simulation::~Simulation()
{
	;
}


} // end namespace nemo
