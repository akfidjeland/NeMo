#ifndef NEMO_NEURON_HPP
#define NEMO_NEURON_HPP

#include <cstddef>
#include <nemo/config.h>
#include "NeuronType.hpp"

namespace nemo {

class Neuron
{
	public :

		Neuron(): a(0), b(0), c(0), d(0), u(0), v(0), sigma(0) {}

		Neuron(const NeuronType&);

		Neuron(const NeuronType&, float fParam[], float fState[]);

		/*! \return i'th parameter of neuron */
		float f_getParameter(size_t i) const;

		/*! \return i'th state variable of neuron */
		float f_getState(size_t i) const;

		/*! set i'th parameter of neuron */
		void f_setParameter(size_t i, float val);

		/*! set i'th state variable of neuron */
		void f_setState(size_t i, float val);

	private :

		float a, b, c, d, u, v, sigma;

		void init(const NeuronType& type);

		void set(float fParam[], float fState[]);

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
