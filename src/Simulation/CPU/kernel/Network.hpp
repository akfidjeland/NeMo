#ifndef NETWORK_HPP
#define NETWORK_HPP

extern "C" {
#include "cpu_kernel.h"
}
#include <vector>


struct NParam {

	NParam(double a, double b, double c, double d) :
		a(a), b(b), c(c), d(d) {}

	double a;
	double b;
	double c;
	double d;
};


struct NState {

	NState(double u, double v, double sigma) :
		u(u), v(v), sigma(sigma) {}

	double u;
	double v;
	double sigma;
};



struct Network {

	Network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[], //set to 0 if not thalamic input required
		unsigned int len);

	std::vector<NParam> param;
	std::vector<NState> state;	

	// last cycle's worth of firing, one entry per neuron
	std::vector<bool_t> fired; 

	// may want to have one rng per neuron or at least per thread
	std::vector<unsigned int> rng; // fixed length: 4
};


#endif
