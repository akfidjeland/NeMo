/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "SimulationBackend.hpp"
#include "fixedpoint.hpp"

namespace nemo {


SimulationBackend::~SimulationBackend()
{
	;
}


const std::vector<unsigned>&
SimulationBackend::step(const std::vector<unsigned>& fstim,
		const current_stimulus& istim)
{
	setFiringStimulus(fstim);
	initCurrentStimulus(istim.size());
	for(current_stimulus::const_iterator i = istim.begin();
			i != istim.end(); ++i) {
		addCurrentStimulus(i->first, i->second);
	}
	finalizeCurrentStimulus(istim.size());
	step();
	return readFiring().neurons;
}


}
