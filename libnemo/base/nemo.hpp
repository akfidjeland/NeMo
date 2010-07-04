#ifndef NEMO_HPP
#define NEMO_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Configuration.hpp"
#include "Network.hpp"
#include "Simulation.hpp"

namespace nemo {

/*! Create a simulation using one of the available backends. Returns NULL if
 * unable to create simulation */
Simulation* simulation(const Network& net, const Configuration& conf);

}

#endif
