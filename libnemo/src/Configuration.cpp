/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Configuration.hpp"
#include "CudaNetwork.hpp"

namespace nemo {

void
Configuration::setStdpFunction(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minWeight,
		float maxWeight)
{
	m_stdpFn = new STDP<float>(prefire, postfire, minWeight, maxWeight);
}



int
Configuration::setCudaDevice(int dev)
{
	return cuda::CudaNetwork::setDevice(dev);
}

} // namespace nemo
