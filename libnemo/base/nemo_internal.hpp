#ifndef NEMO_INTERNAL_HPP
#define NEMO_INTERNAL_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo.hpp>

#include <ConfigurationImpl.hpp>
#include <SimulationBackend.hpp>
#include <NetworkImpl.hpp>

namespace nemo {

SimulationBackend*
simulationBackend(const NetworkImpl& net, const ConfigurationImpl& conf);

}

#endif
