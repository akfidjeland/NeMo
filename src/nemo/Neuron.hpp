#ifndef NEMO_NEURON_HPP
#define NEMO_NEURON_HPP

#include <cstddef>
#include <vector>

#include <nemo/config.h>
#include "NeuronType.hpp"

namespace nemo {

class NEMO_BASE_DLL_PUBLIC Neuron
{
	public :

		Neuron() { }

		explicit Neuron(const NeuronType&);

		Neuron(const NeuronType&, float fParam[], float fState[]);

		/*! \return all parameters of neuron (or NULL if there are none) */
		const float* f_getParameters() const {
			return mf_param.empty() ? NULL : &mf_param[0];
		}

		/*! \return all state variables of neuron (or NULL if there are none) */
		const float* f_getState() const {
			return mf_state.empty() ? NULL : &mf_state[0];
		}

		/*! \return i'th parameter of neuron */
		float f_getParameter(size_t i) const;

		/*! \return i'th state variable of neuron */
		float f_getState(size_t i) const;

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
