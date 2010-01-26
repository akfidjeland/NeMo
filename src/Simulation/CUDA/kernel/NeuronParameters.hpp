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

	private:

		size_t m_partitionSize;

		typedef nemo::Neuron<float> neuron_t;
		typedef std::map<nidx_t, neuron_t> acc_t;
		acc_t m_acc;

		boost::shared_ptr<float> md_arr;  // device data

		nidx_t maxNeuronIdx() const;
		size_t partitionCount() const;
};

#endif
