#define BOOST_TEST_MODULE nemo test_c_api

#include <utility>
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>

#include <nemo.hpp>
#include <nemo.h>

#include "utils.hpp"

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

void
addExcitatoryNeuron(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float a = 0.02f;
	float b = 0.2f;
	float r1 = float(param());
	float r2 = float(param());
	float c = v + 15.0f * r1 * r1;
	float d = 8.0f - 6.0f * r2 * r2;
	float u = b * v;
	float sigma = 5.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
	nemo_add_neuron(c_net, nidx, a, b, c, d, u, v, sigma);
}


void
addExcitatorySynapses(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned source,
		unsigned ncount,
		unsigned scount,
		uirng_t& rtarget,
		urng_t& rweight)
{
	for(unsigned s = 0; s < scount; ++s) {
		unsigned target = rtarget();
		float weight = 0.5f * float(rweight());
		net->addSynapse(source, target, 1U, weight, 0);
		nemo_add_synapse(c_net, source, target, 1U, weight, 0, NULL);
	}
}


void
addInhibitoryNeuron(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned nidx,
		urng_t& param)
{
	float v = -65.0f;
	float r1 = float(param());
	float a = 0.02f + 0.08f * r1;
	float r2 = float(param());
	float b = 0.25f - 0.05f * r2;
	float c = v;
	float d = 2.0f;
	float u = b * v;
	float sigma = 2.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
	nemo_add_neuron(c_net, nidx, a, b, c, d, u, v, sigma);
}



void
addInhibitorySynapses(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned source,
		unsigned ncount,
		unsigned scount,
		uirng_t& rtarget,
		urng_t& rweight)
{
	for(unsigned s = 0; s < scount; ++s) {
		unsigned target = rtarget();
		float weight = float(-rweight());
		net->addSynapse(source, target, 1U, weight, 0);
		nemo_add_synapse(c_net, source, target, 1U, weight, 0, NULL);
	}
}


void
c_runSimulation(
		const nemo_network_t net,
		nemo_configuration_t conf,
		unsigned seconds,
		std::vector<unsigned>& fstim,
		std::vector<unsigned>& istim_nidx,
		std::vector<float>& istim_current,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx)
{
	nemo_simulation_t sim = nemo_new_simulation(net, conf);

	fcycles->clear();
	fnidx->clear();

	//! todo vary the step size between reads to firing buffer
	
	for(unsigned s = 0; s < seconds; ++s)
	for(unsigned ms = 0; ms < 1000; ++ms) {

		unsigned *fired;
		size_t fired_len;

		if(s == 0 && ms == 0) {
			nemo_step(sim, &fstim[0], fstim.size(),
					&istim_nidx[0], &istim_current[0], istim_nidx.size(),
					&fired, &fired_len);
		} else {
			nemo_step(sim, NULL, 0, NULL, NULL, 0, &fired, &fired_len);
		}

		// read back a few synapses every now and then just to make sure it works
		if(ms % 100 == 0) {
			std::vector<synapse_id> synapses = synapseIds(1, 1000);
			float* weights;
			nemo_get_weights(sim, &synapses[0], synapses.size(), &weights);

			unsigned* targets;
			nemo_get_targets(sim, &synapses[0], synapses.size(), &targets);

			unsigned* delays;
			nemo_get_delays(sim, &synapses[0], synapses.size(), &delays);

			unsigned char* plastic;
			nemo_get_plastic(sim, &synapses[0], synapses.size(), &plastic);
		}

		// read back a some membrane potential, just to make sure it works
		if(ms % 100 == 0) {
			float v;
			nemo_get_membrane_potential(sim, 40, &v);
			nemo_get_membrane_potential(sim, 50, &v);
		}

		// push data back onto local buffers
		std::copy(fired, fired + fired_len, back_inserter(*fnidx));
		std::fill_n(back_inserter(*fcycles), fired_len, s*1000 + ms);
	}

	nemo_delete_simulation(sim);
}



void
c_safeCall(nemo_status_t err)
{
	if(err != NEMO_OK) {
		std::cerr << nemo_strerror() << std::endl;
		exit(-1);
	}
}



/* Compare simulation runs using C and C++ APIs with optional firing and
 * current stimulus during cycle 100 */
void
runComparison(bool useFstim, bool useIstim)
{
	unsigned ncount = 1000;
	unsigned scount = 1000;
	//! \todo run test with stdp enabled as well
	bool stdp = false;

	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomTarget(rng, boost::uniform_int<>(0, ncount-1));

	std::cerr << "Nemo version: " << nemo_version() << std::endl;

	std::cerr << "Creating network (C++ API)\n";
	nemo::Network* net = new nemo::Network();
	std::cerr << "Creating network (C API)\n";
	nemo_network_t c_net = nemo_new_network();

	std::cerr << "Populating networks\n";
	for(unsigned nidx=0; nidx < ncount; ++nidx) {
		if(nidx < (ncount * 4) / 5) { // excitatory
			addExcitatoryNeuron(net, c_net, nidx, randomParameter);
			addExcitatorySynapses(net, c_net, nidx, ncount, scount, randomTarget, randomParameter);
		} else { // inhibitory
			addInhibitoryNeuron(net, c_net, nidx, randomParameter);
			addInhibitorySynapses(net, c_net, nidx, ncount, scount, randomTarget, randomParameter);
		}
	}

	nemo::Configuration conf;

	unsigned duration = 2;

	std::vector<unsigned> cycles1, cycles2, nidx1, nidx2;

	std::vector<unsigned> fstim;
	if(useFstim) {
		fstim.push_back(100);
	}

	std::vector< std::pair<unsigned, float> > istim;
	std::vector<unsigned> istim_nidx;
	std::vector<float> istim_current;
	if(useIstim) {
		istim.push_back(std::make_pair(20U, 20.0f));
		istim_nidx.push_back(20);
		istim_current.push_back(20.0f);

		istim.push_back(std::make_pair(40U, 20.0f));
		istim_nidx.push_back(40);
		istim_current.push_back(20.0f);

		istim.push_back(std::make_pair(60U, 20.0f));
		istim_nidx.push_back(60);
		istim_current.push_back(20.0f);
	}

	std::cerr << "Running network (C++ API)\n";
	runSimulation(net, conf, duration, &cycles1, &nidx1, stdp, fstim, istim);

	unsigned cuda_dcount;
	c_safeCall(nemo_cuda_device_count(&cuda_dcount));
	std::cerr << cuda_dcount << " CUDA devices available\n";

	for(unsigned i=0; i < cuda_dcount; ++i) {
		const char* cuda_descr;
		c_safeCall(nemo_cuda_device_description(i, &cuda_descr));
		std::cerr << "\tDevice " << i << ": " << cuda_descr << "\n";
	}


	nemo_configuration_t c_conf = nemo_new_configuration();
	const char* descr;
	c_safeCall(nemo_backend_description(c_conf, &descr));
	std::cerr << descr << std::endl;
	std::cerr << "Running network (C API)\n";
	c_runSimulation(c_net, c_conf, duration,
			fstim, istim_nidx, istim_current, &cycles2, &nidx2);
	std::cerr << "Checking results\n";
	compareSimulationResults(cycles1, nidx1, cycles2, nidx2);

	nemo_delete_configuration(c_conf);
	nemo_delete_network(c_net);
}



BOOST_AUTO_TEST_SUITE(comparison)
	BOOST_AUTO_TEST_CASE(nostim) {
		runComparison(false, false);
	}

	BOOST_AUTO_TEST_CASE(fstim) {
		runComparison(true, false);
	}

	BOOST_AUTO_TEST_CASE(istim) {
		runComparison(false, true);
	}
BOOST_AUTO_TEST_SUITE_END()



/* Test that
 *
 * 1. we get back synapse id
 * 2. the synapse ids are correct (based on our knowledge of the internals
 */
void
testSynapseId()
{
	nemo_network_t net = nemo_new_network();

	synapse_id id00, id01, id10;

	nemo_add_synapse(net, 0, 1, 1, 0.0f, 0, &id00);
	nemo_add_synapse(net, 0, 1, 1, 0.0f, 0, &id01);
	nemo_add_synapse(net, 1, 0, 1, 0.0f, 0, &id10);

	BOOST_REQUIRE_EQUAL(id01 - id00, 1U);
	BOOST_REQUIRE_EQUAL(id10 & 0xffffffff, 0U);
	BOOST_REQUIRE_EQUAL(id00 & 0xffffffff, 0U);
	BOOST_REQUIRE_EQUAL(id01 & 0xffffffff, 1U);
	BOOST_REQUIRE_EQUAL(id00 >> 32, 0U);
	BOOST_REQUIRE_EQUAL(id01 >> 32, 0U);
	BOOST_REQUIRE_EQUAL(id10 >> 32, 1U);

	nemo_delete_network(net);
}

BOOST_AUTO_TEST_CASE(synapse_ids)
{
	testSynapseId();
}
