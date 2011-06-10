#include "Simulation.hpp"

#include <cmath>
#include <algorithm>


#ifdef NEMO_CPU_MULTITHREADED
#include <boost/thread.hpp>
#endif
#include <boost/format.hpp>

#include <nemo/internals.hpp>
#include <nemo/exception.hpp>
#include <nemo/bitops.h>
#include <nemo/fixedpoint.hpp>

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
	/* Creating the neuron population also creates a mapping from global to
	 * (dense) local neuron indices */
	//! \todo create separate collections for each neuron type
	m_neurons(net, 0),
	m_mapper(m_neurons.mapper()),
	m_neuronCount(net.neuronCount()),
	m_fired(m_neuronCount, 0),
	m_recentFiring(m_neuronCount, 0),
	m_delays(m_neuronCount, 0),
	m_cm(net, conf, m_mapper),
	m_current(m_neuronCount, 0)
{
	//! \todo can we finalize cm right away?
	m_cm.finalize(m_mapper, true); // all valid neuron indices are known. See CM ctor.
	for(size_t source=0; source < m_neuronCount; ++source) {
		m_delays[source] = m_cm.delayBits(source);
	}
#ifdef NEMO_CPU_MULTITHREADED
	initWorkers(m_neuronCount, conf.cpuThreadCount());
#endif
	resetTimer();
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
Simulation::fire()
{
	const current_vector_t& current = deliverSpikes();
	update(current);
	setFiring();
	m_timer.step();
}



void
Simulation::setFiringStimulus(const std::vector<unsigned>& fstim)
{
	m_neurons.setFiringStimulus(fstim);
}



void
Simulation::initCurrentStimulus(size_t count)
{
	/* The current is cleared after use, so no need to reset */
}



void
Simulation::addCurrentStimulus(nidx_t neuron, float current)
{
	m_current[m_mapper.localIdx(neuron)] = fx_toFix(current, getFractionalBits());
}



void
Simulation::finalizeCurrentStimulus(size_t count)
{
	/* The current is cleared after use, so no need to reset */
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
	m_neurons.update(start, end, getFractionalBits(),
			&m_current[0], &m_recentFiring[0], &m_fired[0]);
}



void
Simulation::update(
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
Simulation::setNeuron(unsigned g_idx, unsigned nargs, const float args[])
{
	m_neurons.set(g_idx, nargs, args);
}



void
Simulation::setNeuronState(unsigned g_idx, unsigned var, float val)
{
	m_neurons.setState(g_idx, var, val);
}



void
Simulation::setNeuronParameter(unsigned g_idx, unsigned parameter, float val)
{
	m_neurons.setParameter(g_idx, parameter, val);
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
		m_current.at(terminal.target) += terminal.weight;
		LOG("c%lu: n%u -> n%u: %+f (delay %u)\n",
				elapsedSimulation(),
				m_mapper.globalIdx(source),
				m_mapper.globalIdx(terminal.target),
				fx_toFloat(terminal.weight, getFractionalBits()), delay);
	}
}



float
Simulation::getNeuronState(unsigned g_idx, unsigned var) const
{
	return m_neurons.getState(g_idx, var);
}



float
Simulation::getNeuronParameter(unsigned g_idx, unsigned param) const
{
	return m_neurons.getParameter(g_idx, param);
}


float
Simulation::getMembranePotential(unsigned g_idx) const
{
	return m_neurons.getMembranePotential(g_idx);
}



const std::vector<synapse_id>&
Simulation::getSynapsesFrom(unsigned neuron)
{
	return m_cm.getSynapsesFrom(neuron);
}



unsigned
Simulation::getSynapseTarget(const synapse_id& synapse) const
{
	return m_cm.getTarget(synapse);
}



unsigned
Simulation::getSynapseDelay(const synapse_id& synapse) const
{
	return m_cm.getDelay(synapse);
}



float
Simulation::getSynapseWeight(const synapse_id& synapse) const
{
	return m_cm.getWeight(synapse);
}



unsigned char
Simulation::getSynapsePlastic(const synapse_id& synapse) const
{
	return m_cm.getPlastic(synapse);
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
