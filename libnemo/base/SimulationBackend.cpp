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


void
SimulationBackend::step(const std::vector<unsigned>& fstim, const std::vector<float>& istim)
{
	setFiringStimulus(fstim);
	setCurrentStimulus(istim);
	step();
}



void
SimulationBackend::setCurrentStimulus(const std::vector<float>& current)
{
	unsigned fbits = getFractionalBits();
	size_t len = current.size();
	/*! \todo allocate this only once */
	std::vector<fix_t> fx_current(len);
	for(size_t i = 0; i < len; ++i) {
		fx_current.at(i) = fx_toFix(current.at(i), fbits);
	}
	setCurrentStimulus(fx_current);
}




}
