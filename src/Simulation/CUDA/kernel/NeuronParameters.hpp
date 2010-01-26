#ifndef NEURON_PARAMETERS_HPP
#define NEURON_PARAMETERS_HPP

//! \file NeuronParameters.hpp

#include <map>
#include <boost/shared_ptr.hpp>

#include <nemo_types.hpp>

class NeuronParameters
{
	public:

		NeuronParameters(size_t partitionSize);

		void addNeuron(nidx_t neuronIndex,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		void setSigma(class ThalamicInput& th) const;

		void moveToDevice();

		float* deviceData() { return md_arr.get(); }

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_allocated; }

		/*! \return number of /words/ in each plane, i.e. for parameter vector */
		size_t d_vectorLength() const { return m_veclen; }

		size_t wordPitch() const { return m_wpitch; }

		size_t partitionCount() const;

	private:

		size_t m_partitionSize;

		typedef nemo::Neuron<float> neuron_t;
		typedef std::map<nidx_t, neuron_t> acc_t;
		acc_t m_acc;

		boost::shared_ptr<float> md_arr;  // device data

		nidx_t maxNeuronIdx() const;

		size_t m_allocated;

		size_t m_wpitch;
		size_t m_veclen;
};

#endif
