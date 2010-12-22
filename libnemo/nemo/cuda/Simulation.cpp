/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Simulation.hpp"

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

#include <boost/format.hpp>

#include <nemo/exception.hpp>
#include <nemo/NetworkImpl.hpp>
#include <nemo/fixedpoint.hpp>

#include "DeviceAssertions.hpp"
#include "bitvector.hpp"
#include "exception.hpp"

#include "device_assert.cu_h"
#include "kernel.cu_h"
#include "kernel.hpp"


namespace nemo {
	namespace cuda {


Simulation::Simulation(
		const nemo::network::Generator& net,
		const nemo::ConfigurationImpl& conf) :
	m_mapper(net, conf.cudaPartitionSize()),
	m_conf(conf),
	m_neurons(net, m_mapper),
	m_cm(net, conf, m_mapper),
	m_lq(m_mapper.partitionCount(), m_mapper.partitionSize()),
	m_recentFiring(m_mapper.partitionCount(), m_mapper.partitionSize(), false),
	m_firingStimulus(m_mapper.partitionCount(), BV_WORD_PITCH, false, true),
	m_currentStimulus(m_mapper.partitionCount(), m_mapper.partitionSize(), true, true),
	m_current(m_mapper.partitionCount(), m_mapper.partitionSize(), false, false),
	m_firingBuffer(m_mapper),
	m_cycleCounters(m_mapper.partitionCount(), conf.stdpFunction()),
	m_deviceAssertions(m_mapper.partitionCount()),
	m_pitch32(0),
	m_pitch64(0),
	m_stdp(conf.stdpFunction()),
	md_fstim(NULL),
	md_istim(NULL)
{
	if(m_stdp) {
		configureStdp();
	}
	setPitch();
	resetTimer();

	//! \todo do m_cm size reporting here as well
	if(conf.loggingEnabled()) {
		std::cout << "\tLocal queue: " << m_lq.allocated() / (1<<20) << "MB\n";
	}
}



Simulation::~Simulation()
{
	finishSimulation();
}




void
Simulation::configureStdp()
{
	std::vector<float> flfn;

	std::copy(m_stdp->prefire().rbegin(), m_stdp->prefire().rend(), std::back_inserter(flfn));
	std::copy(m_stdp->postfire().begin(), m_stdp->postfire().end(), std::back_inserter(flfn));

	std::vector<fix_t> fxfn(flfn.size());
	unsigned fb = m_cm.fractionalBits();
	for(unsigned i=0; i < fxfn.size(); ++i) {
		fxfn.at(i) = fx_toFix(flfn[i], fb);
	}
	CUDA_SAFE_CALL(
		::configureStdp(
			m_stdp->prefire().size(),
			m_stdp->postfire().size(),
			m_stdp->potentiationBits(),
			m_stdp->depressionBits(),
			const_cast<fix_t*>(&fxfn[0])));
}



void
Simulation::setFiringStimulus(const std::vector<unsigned>& nidx)
{
	if(nidx.empty()) {
		md_fstim = NULL;
		return;
	}

	//! \todo use internal host buffer with pinned memory instead
	size_t pitch = m_firingStimulus.wordPitch();
	std::vector<uint32_t> hostArray(m_firingStimulus.size(), 0);

	for(std::vector<unsigned>::const_iterator i = nidx.begin();
			i != nidx.end(); ++i) {
		//! \todo should check that this neuron exists
		DeviceIdx dev = m_mapper.deviceIdx(*i);
		size_t word = dev.partition * pitch + dev.neuron / 32;
		size_t bit = dev.neuron % 32;
		hostArray[word] |= 1 << bit;
	}

	memcpyToDevice(m_firingStimulus.deviceData(), hostArray, m_mapper.partitionCount() * pitch);
	md_fstim = m_firingStimulus.deviceData();
}



void
Simulation::clearFiringStimulus()
{
	md_fstim = NULL;
}



void
Simulation::initCurrentStimulus(size_t count)
{
	if(count > 0) {
		m_currentStimulus.fill(0);
	}
}



void
Simulation::addCurrentStimulus(nidx_t neuron, float current)
{
	DeviceIdx dev = m_mapper.deviceIdx(neuron);
	fix_t fx_current = fx_toFix(current, m_cm.fractionalBits());
	m_currentStimulus.setNeuron(dev.partition, dev.neuron, fx_current);
}



void
Simulation::finalizeCurrentStimulus(size_t count)
{
	if(count > 0) {
		m_currentStimulus.copyToDevice();
		md_istim = m_currentStimulus.deviceData();
	} else {
		md_istim = NULL;
	}
}



void
Simulation::setCurrentStimulus(const std::vector<fix_t>& current)
{
	if(current.empty()) {
		md_istim = NULL;
		return;
	}
	m_currentStimulus.set(current);
	m_currentStimulus.copyToDevice();
	md_istim = m_currentStimulus.deviceData();
}



void
checkPitch(size_t expected, size_t found)
{
	if(expected != found) {
		std::ostringstream msg;
		msg << "Simulation::checkPitch: pitch mismatch in device memory allocation. "
			"Found " << found << ", expected " << expected << std::endl;
		throw nemo::exception(NEMO_CUDA_MEMORY_ERROR, msg.str());
	}
}


size_t
Simulation::d_allocated() const
{
	return m_firingStimulus.d_allocated()
		+ m_currentStimulus.d_allocated()
		+ m_recentFiring.d_allocated()
		+ m_neurons.d_allocated()
		+ m_firingBuffer.d_allocated()
		+ m_cm.d_allocated();
}


/* Set common pitch and check that all relevant arrays have the same pitch. The
 * kernel uses a single pitch for all 32-bit data */ 
void
Simulation::setPitch()
{
	size_t pitch1 = m_firingStimulus.wordPitch();
	m_pitch32 = m_neurons.wordPitch();
	m_pitch64 = m_recentFiring.wordPitch();
	checkPitch(m_pitch32, m_currentStimulus.wordPitch());
	checkPitch(m_pitch64, m_cm.delayBits().wordPitch());
	checkPitch(pitch1, m_firingBuffer.wordPitch());
	CUDA_SAFE_CALL(nv_setPitch32(m_pitch32));
	CUDA_SAFE_CALL(nv_setPitch64(m_pitch64));
	CUDA_SAFE_CALL(bv_setPitch(pitch1));
}



//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------



void
Simulation::update()
{
	using boost::format;

	m_timer.step();
	m_neurons.step(m_timer.elapsedSimulation());
	initLog();

	::gather(
			m_mapper.partitionCount(),
			m_neurons.rngEnabled(),
			m_timer.elapsedSimulation(),
			m_neurons.df_parameters(),
			m_neurons.du_state(),
			md_istim,
			m_current.deviceData(),
			m_cm.d_fcm(),
			m_cm.d_gqData(),
			m_cm.d_gqFill());

	//! \todo perform copy here

	::stepSimulation(
			m_mapper.partitionCount(),
			m_stdp,
			m_timer.elapsedSimulation(),
			m_recentFiring.deviceData(),
			m_neurons.df_parameters(),
			m_neurons.df_state(),
			md_fstim,
			m_current.deviceData(),
			m_firingBuffer.d_buffer(),
			m_cm.d_outgoingAddr(),
			m_cm.d_outgoing(),
			m_cm.d_gqData(),
			m_cm.d_gqFill(),
			// local spike delivery
			m_lq.d_data(),
			m_lq.d_fill(),
			m_cm.delayBits().deviceData(),
			// cycle counting
			m_cycleCounters.data(),
			m_cycleCounters.pitch());

	/* Get error status before checking assertions, as getting assertion data
	 * may otherwise mask errors here */
	cudaError_t status = cudaGetLastError();

	/* Check device assertions before /reporting/ errors. If we have an
	 * assertion failure we're likely to also have an error, but we'd like to
	 * know what the cause of it was. */
	m_deviceAssertions.check(m_timer.elapsedSimulation());

	if(status != cudaSuccess) {
		throw nemo::exception(NEMO_CUDA_INVOCATION_ERROR,
				str(format("Cuda error in cycle %u: %s")
					% m_timer.elapsedSimulation()
					% cudaGetErrorString(status)));
	}

	m_firingBuffer.sync();

	/* Must clear stimulus pointers in case the low-level interface is used and
	 * the user does not provide any fresh stimulus */
	clearFiringStimulus();

	flushLog();
	endLog();
}



void
Simulation::applyStdp(float reward)
{
	if(!m_stdp) {
		throw exception(NEMO_LOGIC_ERROR, "applyStdp called, but no STDP model specified");
		return;
	}

	if(reward == 0.0f) {
		m_cm.clearStdpAccumulator();
	} else  {
		initLog();
		::applyStdp(
				m_cycleCounters.dataApplySTDP(),
				m_cycleCounters.pitchApplySTDP(),
				m_mapper.partitionCount(),
				m_cm.fractionalBits(),
				m_cm.d_fcm(),
				m_stdp->maxWeight(),
				m_stdp->minWeight(),
				reward);
		flushLog();
		endLog();
	}

	m_deviceAssertions.check(m_timer.elapsedSimulation());
}



void
Simulation::setNeuron(unsigned g_idx,
			float a, float b, float c, float d,
			float u, float v, float sigma)
{
	DeviceIdx d_idx = m_mapper.deviceIdx(g_idx);
	m_neurons.setParameter(d_idx, PARAM_A, a);
	m_neurons.setParameter(d_idx, PARAM_B, b);
	m_neurons.setParameter(d_idx, PARAM_C, c);
	m_neurons.setParameter(d_idx, PARAM_D, d);
	m_neurons.setParameter(d_idx, PARAM_SIGMA, sigma);
	m_neurons.setState(d_idx, STATE_V, v);
	m_neurons.setState(d_idx, STATE_U, u);
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
	return m_cm.getWeights(elapsedSimulation(), synapses);
}


const std::vector<unsigned char>&
Simulation::getPlastic(const std::vector<synapse_id>& synapses)
{
	return m_cm.getPlastic(synapses);
}



FiredList
Simulation::readFiring()
{
	return m_firingBuffer.readFiring();
}



float
Simulation::getMembranePotential(unsigned neuron) const
{
	return m_neurons.getState(m_mapper.deviceIdx(neuron), STATE_V);
}


void
Simulation::finishSimulation()
{
	//! \todo perhaps clear device data here instead of in dtor
	if(m_conf.loggingEnabled()) {
		m_cycleCounters.printCounters(std::cout);
		//! \todo add time summary
	}
}



unsigned long
Simulation::elapsedWallclock() const
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
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
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	m_timer.reset();
}


unsigned
Simulation::getFractionalBits() const
{
	return m_cm.fractionalBits();
}


	} // end namespace cuda
} // end namespace nemo
