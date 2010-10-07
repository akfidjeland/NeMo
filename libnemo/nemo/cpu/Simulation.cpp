#include "Simulation.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>


#ifdef NEMO_CPU_MULTITHREADED
#include <boost/thread.hpp>
#endif

#include <nemo/internals.hpp>
#include <nemo/exception.hpp>
#include <nemo/bitops.h>
#include <nemo/fixedpoint.hpp>

#define SUBSTEPS 4
#define SUBSTEP_MULT 0.25

#ifdef NEMO_CPU_DEBUG_TRACE

#include <cstdio>
#include <cstdlib>

#define LOG(...) fprintf(stdout, __VA_ARGS__);

#else

#define LOG(...)

#endif


namespace nemo {
	namespace cpu {


Simulation::Simulation(
		const nemo::network::Generator& net,
		const nemo::ConfigurationImpl& conf) :
	m_mapper(net),
	//! \todo remove redundant member?
	m_neuronCount(m_mapper.neuronCount()),
	m_a(m_neuronCount, 0),
	m_b(m_neuronCount, 0),
	m_c(m_neuronCount, 0),
	m_d(m_neuronCount, 0),
	m_u(m_neuronCount, 0),
	m_v(m_neuronCount, 0),
	m_sigma(m_neuronCount, 0),
	m_valid(m_neuronCount, false),
	m_fired(m_neuronCount, 0),
	m_recentFiring(m_neuronCount, 0),
	m_delays(m_neuronCount, 0),
	//! \todo Determine fixedpoint format based on input network
	m_cm(net, conf, m_mapper),
	m_current(m_neuronCount, 0),
	m_fstim(m_neuronCount, 0),
	m_rng(m_neuronCount)
{
	nemo::initialiseRng(m_mapper.minLocalIdx(), m_mapper.maxLocalIdx(), m_rng);
	setNeuronParameters(net, m_mapper);
	m_cm.finalize(m_mapper); // all valid neuron indices are known. See CM ctor.
	for(size_t source=0; source < m_neuronCount; ++source) {
		m_delays[source] = m_cm.delayBits(source);
	}
#ifdef NEMO_CPU_MULTITHREADED
	initWorkers(m_neuronCount, conf.cpuThreadCount());
#endif
	resetTimer();
}


void
Simulation::setNeuronParameters(
		const nemo::network::Generator& net,
		Mapper& mapper)
{
	using namespace nemo::network;

	for(neuron_iterator i = net.neuron_begin(), i_end = net.neuron_end();
			i != i_end; ++i) {
		nidx_t nidx = mapper.addGlobal((*i).first);
		const Neuron<float>& n = i->second;
		m_a.at(nidx) = n.a;	
		m_b.at(nidx) = n.b;	
		m_c.at(nidx) = n.c;	
		m_d.at(nidx) = n.d;	
		m_u.at(nidx) = n.u;	
		m_v.at(nidx) = n.v;	
		m_sigma.at(nidx) = n.sigma;	
		m_valid.at(nidx) = true;
	}
}


#ifdef NEMO_CPU_MULTITHREADED

/* Allocate work to each thread */
void
Simulation::initWorkers(size_t neurons, unsigned threads)
{
	//! \todo log level of hardware concurrency
	size_t jobSize = neurons / threads;
	for(unsigned t=0; t < threads; ++t) {
		m_workers.push_back(Worker(t, jobSize, neurons, this));
	}
}

#endif



unsigned
Simulation::getFractionalBits() const
{
	return m_cm.fractionalBits();
}



void
Simulation::step()
{
	const current_vector_t& current = deliverSpikes();
	update(m_fstim, current);
	setFiring();
	m_timer.step();
	//! \todo add separate unset so as to only touch minimal number of words
	//each iteration. Alternatively, just unset after use.
	std::fill(m_fstim.begin(), m_fstim.end(), 0);
}



void
Simulation::setFiringStimulus(const std::vector<unsigned>& fstim)
{
	// m_fstim should already be clear at this point
	for(std::vector<unsigned>::const_iterator i = fstim.begin();
			i != fstim.end(); ++i) {
		m_fstim.at(m_mapper.localIdx(*i)) = true;
	}
}



void
Simulation::setCurrentStimulus(const std::vector<fix_t>& current)
{
	if(current.empty()) {
		//! do we need to clear current?
		return;
	}
	/*! \todo We need to deal with the mapping from global to local neuron
	 * indices. Before doing this, we should probably change the interface
	 * here. Note that this function is only used internally (see mpi::Worker),
	 * so we might be able to use the existing interface, and make sure that we
	 * only use local indices. */
	throw nemo::exception(NEMO_API_UNSUPPORTED, "setting current stimulus vector not supported for CPU backend");
#if 0
	if(current.size() != m_current.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "current stimulus vector not of expected size");
	}
	m_current = current;
#endif
}



void
Simulation::updateRange(int start, int end)
{
	unsigned fbits = getFractionalBits();

	for(int n=start; n < end; n++) {

		if(!m_valid[n]) {
			continue;
		}

		float current = fx_toFloat(m_current[n], fbits);
		m_current[n] = 0;

		if(m_sigma[n] != 0.0f) {
			current += m_sigma[n] * (float) m_rng[n].gaussian();
		}

		m_fired[n] = 0;

		for(unsigned int t=0; t<SUBSTEPS; ++t) {
			if(!m_fired[n]) {
				m_v[n] += SUBSTEP_MULT * ((0.04* m_v[n] + 5.0) * m_v[n] + 140.0 - m_u[n] + current);
				/*! \todo: could pre-multiply this with a, when initialising memory */
				m_u[n] += SUBSTEP_MULT * (m_a[n] * (m_b[n] * m_v[n] - m_u[n]));
				m_fired[n] = m_v[n] >= 30.0;
			}
		}

		m_fired[n] |= m_fstim[n];
		m_recentFiring[n] = (m_recentFiring[n] << 1) | (uint64_t) m_fired[n];

		if(m_fired[n]) {
			m_v[n] = m_c[n];
			m_u[n] += m_d[n];
			LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}

	}
}



void
Simulation::update(
		const stimulus_vector_t& fstim,
		const current_vector_t& current)
{
#ifdef NEMO_CPU_MULTITHREADED
	if(m_workers.size() > 1) {
		/* It's possible to reduce thread creation overheads here by creating
		 * threads in the nemo::Simulation ctor and send signals to activate the
		 * threads. However, this was found to not produce any measurable speedup
		 * over this simpler implementation */
		boost::thread_group threads;
		for(std::vector<Worker>::const_iterator i = m_workers.begin();
				i != m_workers.end(); ++i) {
			threads.create_thread(*i);
		}
		/* All threads work here, filling in different part of the simulation data */
		threads.join_all();
	} else
#else
		updateRange(0, m_neuronCount);
#endif

	m_cm.accumulateStdp(m_recentFiring);
}



//! \todo use per-thread buffers and just copy these in bulk
void
Simulation::setFiring()
{
	m_firingBuffer.enqueueCycle();
	for(unsigned n=0; n < m_neuronCount; ++n) { 
		if(m_fired[n]) {
			m_firingBuffer.addFiredNeuron(m_mapper.globalIdx(n));
		}
	}
}



FiredList
Simulation::readFiring()
{
	return m_firingBuffer.dequeueCycle();
}



void
Simulation::applyStdp(float reward)
{
	m_cm.applyStdp(reward);
}



Simulation::current_vector_t&
Simulation::deliverSpikes()
{
	/* Ignore spikes outside of max delay. We keep these older spikes as they
	 * may be needed for STDP */
	uint64_t validSpikes = ~(((uint64_t) (~0)) << m_cm.maxDelay());

	for(size_t source=0; source < m_neuronCount; ++source) {

		uint64_t f = m_recentFiring[source] & validSpikes & m_delays[source];

		int delay = 0;
		while(f) {
			int shift = 1 + ctz64(f);
			delay += shift;
			f = f >> shift;
			deliverSpikesOne(source, delay);
		}
	}

	return m_current;
}



void
Simulation::deliverSpikesOne(nidx_t source, delay_t delay)
{
	const nemo::Row& row = m_cm.getRow(source, delay);

	for(unsigned s=0; s < row.len; ++s) {
		const FAxonTerminal& terminal = row[s];
		assert(terminal.target < m_current.size());
		m_current.at(terminal.target) += terminal.weight;
		LOG("c%lu: n%u -> n%u: %+f (delay %u)\n",
				elapsedSimulation(),
				m_mapper.globalIdx(source),
				m_mapper.globalIdx(terminal.target),
				fx_toFloat(terminal.weight, getFractionalBits()), delay);
	}
}



const std::vector<unsigned>&
Simulation::getTargets(const std::vector<synapse_id>& synapses)
{
	return m_cm.getTargets(synapses);
}



const std::vector<unsigned>&
Simulation::getDelays(const std::vector<synapse_id>& synapses)
{
	return m_cm.getDelays(synapses);
}



const std::vector<float>&
Simulation::getWeights(const std::vector<synapse_id>& synapses)
{
	return m_cm.getWeights(synapses);
}



const std::vector<unsigned char>&
Simulation::getPlastic(const std::vector<synapse_id>& synapses)
{
	return m_cm.getPlastic(synapses);
}



unsigned long
Simulation::elapsedWallclock() const
{
	return m_timer.elapsedWallclock();
}



unsigned long
Simulation::elapsedSimulation() const
{
	return m_timer.elapsedSimulation();
}



void
Simulation::resetTimer()
{
	m_timer.reset();
}



void
chooseHardwareConfiguration(nemo::ConfigurationImpl& conf, int threadCount)
{
	conf.setBackend(NEMO_BACKEND_CPU);
	/*! \todo get processor name */
#ifdef NEMO_CPU_MULTITHREADED
	if(threadCount < 1) {
		conf.setCpuThreadCount(std::max(1U, boost::thread::hardware_concurrency()));
	} else {
		conf.setCpuThreadCount(threadCount);
	}
#else
	if(threadCount > 1) {
		throw nemo::exception(NEMO_INVALID_INPUT, "nemo compiled without multithreading support.");
	} else {
		conf.setCpuThreadCount(1);
	}
#endif
}


	} // namespace cpu
} // namespace nemo
