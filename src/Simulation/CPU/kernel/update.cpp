extern "C" {
#include "cpu_kernel.h"
}

#include "Network.hpp"
#include <cmath>

//#define VERBOSE


//! \todo move to Network.hpp
struct Network*
set_network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[], //set to 0 if not thalamic input required
		unsigned int ncount,
		delay_t maxDelay)
{
	return new Network(a, b, c, d, u, v, sigma, ncount, maxDelay);
}


void
delete_network(Network* net)
{
	delete net; 
}



//! \todo move into Network class
bool_t*
update(Network* network, unsigned int fstim[], double current2[])
{
	network->step(fstim, current2);
}
