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

namespace nemo {

Simulation*
Simulation::create(bool setReverse)
{
	int dev = cuda::CudaNetwork::selectDevice();
	return dev == -1 ? NULL : new cuda::CudaNetwork(setReverse);
}



Simulation::~Simulation()
{
	;
}

} // end namespace nemo
