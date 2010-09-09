#include <vector>
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>
#include <nemo.hpp>

#include <cmath>

void
runSimulation(
		const nemo::Network* net,
		nemo::Configuration conf,
		unsigned seconds,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx,
		bool stdp,
		std::vector<unsigned> initStimulus)
{
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	fcycles->clear();
	fnidx->clear();

	std::vector<unsigned> noStimulus;

	//! todo vary the step size between reads to firing buffer
	
	for(unsigned s = 0; s < seconds; ++s) {
		for(unsigned ms = 0; ms < 1000; ++ms) {

			const std::vector<unsigned>& fired = sim->step((s == 0 && ms == 0) ? initStimulus : noStimulus);

			std::copy(fired.begin(), fired.end(), back_inserter(*fnidx));
			std::fill_n(back_inserter(*fcycles), fired.size(), s*1000 + ms);
		}
		if(stdp) {
			sim->applyStdp(1.0);
		}
	}

	BOOST_CHECK_EQUAL(fcycles->size(), fnidx->size());
}



void
compareSimulationResults(
		const std::vector<unsigned>& cycles1,
		const std::vector<unsigned>& nidx1,
		const std::vector<unsigned>& cycles2,
		const std::vector<unsigned>& nidx2)
{
	BOOST_CHECK_EQUAL(nidx1.size(), nidx2.size());
	BOOST_CHECK_EQUAL(cycles1.size(), cycles2.size());

	for(size_t i = 0; i < cycles1.size(); ++i) {
		// no point continuing after first divergence, it's only going to make
		// output hard to read.
		BOOST_CHECK_EQUAL(cycles1.at(i), cycles2.at(i));
		BOOST_CHECK_EQUAL(nidx1.at(i), nidx2.at(i));
		if(nidx1.at(i) != nidx2.at(i)) {
			BOOST_FAIL("c" << cycles1.at(i) << "/" << cycles2.at(i));
		}
	}
}


void
setBackend(backend_t backend, nemo::Configuration& conf)
{
	switch(backend) {
		case NEMO_BACKEND_CPU: conf.setCpuBackend(); break;
		case NEMO_BACKEND_CUDA: conf.setCudaBackend(); break;
		default: BOOST_REQUIRE(false);
	}
}


nemo::Configuration
configuration(bool stdp, unsigned partitionSize,
		backend_t backend = NEMO_BACKEND_CUDA)
{
	nemo::Configuration conf;

	if(stdp) {
		std::vector<float> pre(20);
		std::vector<float> post(20);
		for(unsigned i = 0; i < 20; ++i) {
			float dt = float(i + 1);
			pre.at(i) = 0.1 * expf(-dt / 20.0f);
			post.at(i) = -0.08 * expf(-dt / 20.0f);
		}
		conf.setStdpFunction(pre, post, -10.0, 10.0);
	}

	setBackend(backend, conf);
	conf.setCudaPartitionSize(partitionSize);

	return conf;
}


void
addExcitatoryNeuron(unsigned nidx, nemo::Network& net, float sigma)
{
	float r = 0.5;
	float b = 0.25f - 0.05f * r;
	float v = -65.0;
	net.addNeuron(nidx, 0.02f + 0.08f * r, b, v, 2.0f, b*v, v, sigma);
}


std::vector<synapse_id>
synapseIds(unsigned neuron, unsigned synapses)
{
	std::vector<synapse_id> ids(synapses);
	for(uint32_t sidx = 0; sidx < synapses; ++sidx) {
		ids[sidx] = (uint64_t(neuron) << 32) | uint64_t(sidx);
	}
	return ids;
}
