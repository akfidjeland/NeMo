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

#include <nemo/Configuration.hpp>
#include <nemo/Network.hpp>
#include <nemo/Simulation.hpp>

namespace nemo {

/*! Create a simulation using one of the available backends. Returns NULL if
 * unable to create simulation.
 *
 * Any missing/unspecified fields in the configuration are filled in */
NEMO_DLL_PUBLIC
Simulation* simulation(const Network& net, Configuration& conf);

}

#endif
