#include "fixedpoint.hpp"

#include <stdexcept>
#include <sstream>


fix_t
fixedPoint(float f, uint fractionalBits)
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
