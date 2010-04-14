/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "nemo.hpp"
#include "CudaNetwork.hpp"
#include "Connectivity.hpp"

namespace nemo {

Simulation*
Simulation::create()
{
	int dev = cuda::CudaNetwork::selectDevice();
	return dev == -1 ? NULL : new cuda::CudaNetwork();
}


Simulation*
Simulation::create(const Connectivity& net, const Configuration& conf)
{
	int dev = cuda::CudaNetwork::selectDevice();
	return dev == -1 ? NULL : new cuda::CudaNetwork(net, conf);
}



Simulation::~Simulation()
{
	;
}

} // end namespace nemo
