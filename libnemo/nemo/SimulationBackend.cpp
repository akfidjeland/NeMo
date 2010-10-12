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


const Simulation::firing_output&
SimulationBackend::step()
{
	initCurrentStimulus(0);
	finalizeCurrentStimulus(0);
	update();
	return readFiring().neurons;
}


const Simulation::firing_output&
SimulationBackend::step(const firing_stimulus& fstim)
{
	setFiringStimulus(fstim);
	initCurrentStimulus(0);
	finalizeCurrentStimulus(0);
	update();
	return readFiring().neurons;
}


const Simulation::firing_output&
SimulationBackend::step(const current_stimulus& istim)
{
	initCurrentStimulus(istim.size());
	for(current_stimulus::const_iterator i = istim.begin();
			i != istim.end(); ++i) {
		addCurrentStimulus(i->first, i->second);
	}
	finalizeCurrentStimulus(istim.size());
	update();
	return readFiring().neurons;
}


const Simulation::firing_output&
SimulationBackend::step(const firing_stimulus& fstim, const current_stimulus& istim)
{
	setFiringStimulus(fstim);
	initCurrentStimulus(istim.size());
	for(current_stimulus::const_iterator i = istim.begin();
			i != istim.end(); ++i) {
		addCurrentStimulus(i->first, i->second);
	}
	finalizeCurrentStimulus(istim.size());
	update();
	return readFiring().neurons;
}


}
