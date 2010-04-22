/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Configuration.hpp"
#include "CudaSimulation.hpp"

namespace nemo {

Configuration::Configuration() :
	m_logging(false),
	m_cudaPartitionSize(cuda::Simulation::defaultPartitionSize()),
	m_cudaFiringBufferLength(cuda::Simulation::defaultFiringBufferLength())
{
	;
}

void
Configuration::setStdpFunction(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minWeight,
		float maxWeight)
{
	m_stdpFn = STDP<float>(prefire, postfire, minWeight, maxWeight);
}



int
Configuration::setCudaDevice(int dev)
{
	return cuda::Simulation::setDevice(dev);
}

} // namespace nemo


std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf)
{
	return o
		<< "STDP=" << conf.stdpFunction().enabled() << " "
		<< "cuda_ps=" << conf.cudaPartitionSize();
	//! \todo print more infor about STDP
}
