#ifndef NEMO_NEURON_HPP
#define NEMO_NEURON_HPP

#include <cstddef>
#include <nemo/config.h>

namespace nemo {

class Neuron
{
	public :

		Neuron(): a(0), b(0), c(0), d(0), u(0), v(0), sigma(0) {}

		/*! Create an Izhikevich neuron */
		Neuron(float a, float b, float c, float d, float u, float v, float sigma) :
			a(a), b(b), c(c), d(d), u(u), v(v), sigma(sigma) {}

		/*! \return i'th parameter of neuron */
		float f_getParameter(size_t i) const;

		/*! \return i'th state variable of neuron */
		float f_getState(size_t i) const;

		float a, b, c, d, u, v, sigma;

	private :

#ifdef NEMO_MPI_ENABLED

		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & a;
			ar & b;
			ar & c;
			ar & d;
			ar & u;
			ar & v;
			ar & sigma;
		}
#endif

};

}


#endif
