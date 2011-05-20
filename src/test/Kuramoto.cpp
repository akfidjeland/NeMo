#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <nemo.hpp>
#include <nemo/util.h>

namespace nemo {
	namespace test {
		namespace kuramoto {


class OscillatorNetwork : public nemo::Network
{
	public :

		OscillatorNetwork() {
			m_type = addNeuronType("Kuramoto");
		}

		void add(unsigned idx, float frequency, float phase) {
			static float args[2];
			args[0] = frequency;
			args[1] = phase;
			addNeuron(m_type, idx, 2, args);
		}

		void connect(unsigned source, unsigned target,
				unsigned lag, float strength) {
			addSynapse(source, target, lag, strength, false);
		}

	private :

		unsigned m_type;
};


void
testUncoupled()
{
	OscillatorNetwork net;

	const unsigned ncount = 1200;
	const float frequency = 0.1f;
	float phase = 0.0f;
	for(unsigned n=0; n<ncount; ++n) {
		net.add(n, frequency, phase);
	}

	Configuration conf;
	boost::scoped_ptr<Simulation> sim(simulation(net, conf));

	const float tolerance = 0.001f; // percent
	const unsigned duration = 1000;
	for(unsigned t=0; t<duration; ++t) {
		sim->step();
		phase = fmodf(phase + frequency, 2*M_PI);
		BOOST_REQUIRE_CLOSE(sim->getNeuronState(ncount/2, 0), phase, tolerance);
	}
}



void
testSimpleCoupled()
{
	OscillatorNetwork net;

	float freq[2] = {0.1f, 0.1f};
	float phase[2] = {0.0f, 0.11f};

	net.add(0, freq[0], phase[0]);
	net.add(1, freq[1], phase[1]);
	net.connect(0, 1, 1, 1.0);

	Configuration conf;
	boost::scoped_ptr<Simulation> sim(simulation(net, conf));

	for(unsigned t=0; t<100; ++t) {
		sim->step();

		float shift = sinf(phase[1]-phase[0]); // weight is 1
		phase[1] += freq[1] + shift;
		phase[0] += freq[0];
		
		for(unsigned i=0; i<net.neuronCount(); ++i) {
			phase[i] = fmodf(phase[i], float(2*M_PI));
			const float tolerance = 0.001f; // percent
			BOOST_REQUIRE_CLOSE(sim->getNeuronState(i,0), phase[i], tolerance);
		}
	}
}




/*! Test n-to-1 coupling
 *
 * Using a large number of oscillators, one of which is coupled with all the
 * others, compare simulation state with expected state
 *
 * There will be eventual divergence due to different floating point
 * implementations.
 *
 * The 'noise' connections are 0-strength couplings, to flush out errors
 * resulting from race conditions.
 */
void
testNto1(unsigned ncount, bool noise)
{
	OscillatorNetwork net;

	unsigned duration = 50; // cycles

	float frequency = 0.1f; // same frequency for all
	float strength = 1.0f / float(ncount);

	float phase0 = 0.0f;
	float phaseN = 0.0f;

	net.add(0, frequency, phase0);
	for(unsigned n=0; n<ncount; ++n) {
		net.add(n+1, frequency, phaseN);
		net.connect(n+1, 0, 1, strength);
		if(noise) {
			for(unsigned tgt=0; tgt<ncount; ++tgt) {
				net.connect(n+1, tgt, 1, 0.0f);
			}
		}
	}

	Configuration conf;
	boost::scoped_ptr<Simulation> sim(simulation(net, conf));


	for(unsigned t=0; t<duration; ++t) {
		sim->step();

		/* Sum of weights is one */
		phase0 += frequency + ncount * strength * sinf(phase0-phaseN);
		phase0 = fmod(phase0, float(2*M_PI));
		phaseN += frequency;
		phaseN = fmod(phaseN, float(2*M_PI));

		const float tolerance = 0.001f; // percent
		BOOST_REQUIRE_CLOSE(sim->getNeuronState(0,0), phase0, tolerance);
		for(unsigned n=0; n<ncount; ++n) {
			BOOST_REQUIRE_CLOSE(sim->getNeuronState(n+1,0), phaseN, tolerance);
		}
	}
}


}	}	} // end namespaces



BOOST_AUTO_TEST_SUITE(kuramoto)
	BOOST_AUTO_TEST_CASE(uncoupled) { nemo::test::kuramoto::testUncoupled(); }
	BOOST_AUTO_TEST_SUITE(coupled)
		BOOST_AUTO_TEST_CASE(onetoone) { nemo::test::kuramoto::testSimpleCoupled(); }
		BOOST_AUTO_TEST_CASE(CtoI) { nemo::test::kuramoto::testNto1(100, false); }
		BOOST_AUTO_TEST_CASE(CtoInoise) { nemo::test::kuramoto::testNto1(100, true); }
		BOOST_AUTO_TEST_CASE(MtoI) { nemo::test::kuramoto::testNto1(1000, false); }
		BOOST_AUTO_TEST_CASE(MtoInoise) { nemo::test::kuramoto::testNto1(1000, true); }
	BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
