#ifndef NEMO_INTERNALS_HPP
#define NEMO_INTERNALS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo.hpp>

#include <nemo/ConfigurationImpl.hpp>
#include <nemo/SimulationBackend.hpp>
#include <nemo/NetworkImpl.hpp>

namespace nemo {

SimulationBackend*
simulationBackend(const NetworkImpl& net, const ConfigurationImpl& conf);

void
setDefaultHardware(nemo::ConfigurationImpl& conf);

void
setCudaDeviceConfiguration(nemo::ConfigurationImpl& conf, int device = -1);

const char*
cudaDeviceDescription(unsigned device);

}

#endif
