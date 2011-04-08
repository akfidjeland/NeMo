/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "SimulationBackend.hpp"

#include <boost/array.hpp>

#include "fixedpoint.hpp"

namespace nemo {


SimulationBackend::~SimulationBackend()
{
	;
}


const Simulation::firing_output&
SimulationBackend::step()
{
	prefire();
	initCurrentStimulus(0);
	finalizeCurrentStimulus(0);
	fire();
	postfire();
	return readFiring().neurons;
}


const Simulation::firing_output&
SimulationBackend::step(const firing_stimulus& fstim)
{
	prefire();
	initCurrentStimulus(0);
	finalizeCurrentStimulus(0);
	setFiringStimulus(fstim);
	fire();
	postfire();
	return readFiring().neurons;
}


const Simulation::firing_output&
SimulationBackend::step(const current_stimulus& istim)
{
	prefire();
	initCurrentStimulus(istim.size());
	for(current_stimulus::const_iterator i = istim.begin();
			i != istim.end(); ++i) {
		addCurrentStimulus(i->first, i->second);
	}
	finalizeCurrentStimulus(istim.size());
	fire();
	postfire();
	return readFiring().neurons;
}


const Simulation::firing_output&
SimulationBackend::step(const firing_stimulus& fstim, const current_stimulus& istim)
{
	prefire();
	initCurrentStimulus(istim.size());
	for(current_stimulus::const_iterator i = istim.begin();
			i != istim.end(); ++i) {
		addCurrentStimulus(i->first, i->second);
	}
	finalizeCurrentStimulus(istim.size());
	setFiringStimulus(fstim);
	fire();
	postfire();
	return readFiring().neurons;
}



void
SimulationBackend::setNeuron(unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	static boost::array<float, 5> param;
	static boost::array<float, 2> state;

	param[0] = a; param[1] = b; param[2] = c; param[3] = d; param[4] = sigma;
	state[0] = u; state[1] = v;
	setNeuron(idx, param.c_array(), state.c_array());
}

}
