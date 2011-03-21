#ifndef NEMO_NEURON_HPP
#define NEMO_NEURON_HPP

#include <cstddef>
#include <vector>

#include <nemo/config.h>
#include "NeuronType.hpp"

namespace nemo {

class Neuron
{
	public :

		Neuron() { }

		explicit Neuron(const NeuronType&);

		Neuron(const NeuronType&, float fParam[], float fState[]);

		/*! \return i'th parameter of neuron */
		float f_getParameter(size_t i) const { return f_paramRef(i); }

		/*! \return i'th state variable of neuron */
		float f_getState(size_t i) const { return f_stateRef(i); }

		/*! set i'th parameter of neuron */
		void f_setParameter(size_t i, float val);

		/*! set i'th state variable of neuron */
		void f_setState(size_t i, float val);

	private :

		void init(const NeuronType& type);

		void set(float fParam[], float fState[]);

		std::vector<float> mf_param;
		std::vector<float> mf_state;

		const float& f_paramRef(size_t i) const;
		const float& f_stateRef(size_t i) const;

#ifdef NEMO_MPI_ENABLED
#	error "MPI serialisation of neuron type is broken"

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
