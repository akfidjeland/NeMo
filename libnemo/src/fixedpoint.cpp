/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "fixedpoint.hpp"

#include <stdexcept>
#include <sstream>


fix_t
fx_toFix(float f, uint fractionalBits)
{
	if(abs(int(f)) >= 1 << (31 - fractionalBits)) {
		std::ostringstream msg;
		msg << "Fixed-point overflow. Value " << f
			<< " does not fit into fixed-point format Q" 
			<< 31-fractionalBits << "." << fractionalBits << std::endl;
		throw std::runtime_error(msg.str());
	}
	return static_cast<fix_t>(f * (1<<fractionalBits));
}



float
fx_toFloat(fix_t v, uint fractionalBits)
{
	return float(v) / float(1<<fractionalBits);
}
