/* Tests for STDP functionality */

#include <cmath>
#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <boost/test/unit_test.hpp>

#include <nemo.hpp>
#include "utils.hpp"

/* The test network consists of two groups of the same size. Connections
 * between these groups are one-way and are organised in a one-to-one fashion.
 */
static const unsigned groupSize = 2 * 768;

/*The initial weight should be too low to induce firing
 * based on just a single spike and still allow a few depressions before
 * reaching zero. */
static const float initWeight = 5.0f; 

static const unsigned nepochs = 4;
static const unsigned preFire = 10; // ms within epoch
static const unsigned postFireDelay = 10; // ms within epoch


float
dwPre(int dt)
{
	assert(dt <= 0);
	return 1.0f * expf(float(dt) / 20.0f);
}



float
dwPost(int dt)
{
	assert(dt >= 0.0f);
	return -0.8f * expf(float(-dt) / 20.0f);
}


nemo::Configuration
configuration(backend_t backend)
{
	nemo::Configuration conf;

	std::vector<float> pre(20);
	std::vector<float> post(20);
	for(unsigned i = 0; i < 20; ++i) {
		int dt = i;
		pre.at(i) = dwPre(-dt);
		post.at(i) = dwPost(dt);
	}
	/* don't allow negative synapses to go much more negative.
	 * This is to avoid having large negative input currents,
	 * which will result in extra firing (by forcing 'u' to
	 * go highly negative) */
	conf.setStdpFunction(pre, post, -0.5, 2*initWeight);
	setBackend(backend, conf);

	return conf;
}



unsigned
globalIdx(unsigned group, unsigned local)
{
	return group * groupSize + local;
}


unsigned
localIdx(unsigned global)
{
	return global % groupSize;
}


/* The synaptic delays between neurons in the two groups depend only on the
 * index of the second neuron */
unsigned
delay(unsigned local)
{
	return 1 + (local % 20);
}


/* Return number of synapses per neuron */
unsigned
construct(nemo::Network& net, bool noiseConnections)
{
	/* Neurons in the two groups have standard parameters and no spontaneous
	 * firing */
	for(unsigned group=0; group < 2; ++group) {
		for(unsigned local=0; local < groupSize; ++local) {
			float r = 0.5;
			float b = 0.25f - 0.05f * r;
			float v = -65.0;
			net.addNeuron(globalIdx(group, local),
					0.02f + 0.08f * r, b, v, 2.0f, b*v, v, 0.0f);
		}
	}

	/* The plastic synapses  are one-way, from group 0 to group 1. The delay
	 * varies depending on the target neuron. The weights are set that a single
	 * spike is enough to induce firing in the postsynaptic neuron. */
	for(unsigned local=0; local < groupSize; ++local) {
		net.addSynapse(
				globalIdx(0, local),
				globalIdx(1, local),
				delay(local),
				initWeight, 1);
	}
	
	/* To complicate spike delivery and STDP computation, add a number of
	 * connections with very low negative weights. Even if potentiated, these
	 * will not lead to additional firing. Use a mix of plastic and static
	 * synapses. */
	if(noiseConnections) {
		for(unsigned lsrc=0; lsrc < groupSize; ++lsrc) 
		for(unsigned ltgt=0; ltgt < groupSize; ++ltgt) {
			if(lsrc != ltgt) {
				net.addSynapse(
						globalIdx(0, lsrc),
						globalIdx(1, ltgt),
						delay(ltgt + lsrc),
						-0.00001f,
						 ltgt & 0x1);
			}
		}
	}

	return noiseConnections ? groupSize : 1;
}



void
stimulateGroup(unsigned group, std::vector<unsigned>& fstim)
{
	for(unsigned local=0; local < groupSize; ++local) {
		fstim.push_back(globalIdx(group, local));
	}
}


/* Neurons are only stimulated at specific times */
const std::vector<unsigned>&
stimulus(unsigned ms, std::vector<unsigned>& fstim)
{
	if(ms == preFire) {
		stimulateGroup(0, fstim);
	} else if(ms == preFire + postFireDelay) {
		stimulateGroup(1, fstim);
	}
	return fstim;
}


void
verifyWeightChange(unsigned epoch, nemo::Simulation* sim, unsigned m)
{
	unsigned checked = 0; 

	for(unsigned local = 0; local < groupSize; ++local) {

		std::vector<synapse_id> synapses = synapseIds(globalIdx(0, local), m);
		const std::vector<unsigned>& targets = sim->getTargets(synapses);
		const std::vector<float>& weights = sim->getWeights(synapses);
		const std::vector<unsigned>& delays = sim->getDelays(synapses);
		const std::vector<unsigned char>& plastic = sim->getPlastic(synapses);

		for(unsigned s = 0; s < targets.size(); ++s) {

			if(local != localIdx(targets.at(s)))
				continue;

			BOOST_REQUIRE_EQUAL(delay(localIdx(targets.at(s))), delays.at(s));
			BOOST_REQUIRE(plastic.at(s));

			/* dt is positive for pre-post pair, and negative for post-pre
			 * pairs */ 
			int dt = -(int(postFireDelay - delays.at(s)));

			float dw_expected = 0.0f; 
			if(dt > 0) {
				dw_expected = dwPost(dt-1);
			} else if(dt <= 0) {
				dw_expected = dwPre(dt);
			}

			float expectedWeight = initWeight + epoch * dw_expected;
			float actualWeight = weights.at(s);

			const float tolerance = 0.001f; // percent
			BOOST_REQUIRE_CLOSE(expectedWeight, actualWeight, tolerance);

			checked += 1;
		}
	}

	std::cout << "Epoch " << epoch << ": checked " << checked << " synapses\n";
}


void
testStdp(backend_t backend, bool noiseConnections)
{
	nemo::Network net;
	unsigned m = construct(net, noiseConnections);
	nemo::Configuration conf = configuration(backend);

	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));

	verifyWeightChange(0, sim.get(), m);

	for(unsigned epoch = 1; epoch <= nepochs; ++epoch) {
		for(unsigned ms = 0; ms < 100; ++ms) {
			std::vector<unsigned> fstim;
			sim->step(stimulus(ms, fstim));
		}
		/* During the preceding epoch each synapse should have
		 * been updated according to the STDP rule exactly
		 * once. The magnitude of the weight change will vary
		 * between synapses according to their delay */
		sim->applyStdp(1.0);
		verifyWeightChange(epoch, sim.get(), m);
	}
}
