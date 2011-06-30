#ifndef NEMO_CPU_NEURONS_HPP
#define NEMO_CPU_NEURONS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/multi_array.hpp>

#include <nemo/RandomMapper.hpp>
#include <nemo/Plugin.hpp>
#include <nemo/RNG.hpp>
#include <nemo/network/Generator.hpp>
#include <nemo/cpu/plugins/neuron_model.h>

namespace nemo {
	namespace cpu {


/*! Neuron population for CPU backend
 *
 * The neurons are stored internally in dense structure-of-arrays with
 * contigous local indices starting from zero.
 *
 * \todo deal with multi-threading inside this class
 */
class Neurons
{
	public :

		/*! Set up local storage for all neurons with the generator neuron type id.
		 *
		 * Doing so also creates a mapping from global to dense local neuron
		 * indices. The resulting mapper can be queried via \a mapper */
		Neurons(const network::Generator& net, unsigned id);

		/*! Update the state of all neurons
		 *
		 * \post the input current vector is set to all zero.
		 * \post the internal firing stimulus buffer (\a m_fstim) is set to all false.
		 */
		void update(int start, int end, unsigned fbits,
			fix_t current[], uint64_t recentFiring[], unsigned fired[]);

		/*! \copydoc nemo::Network::getNeuronState */
		float getState(unsigned g_idx, unsigned var) const {
			return m_state[0][stateIndex(var)][m_mapper.localIdx(g_idx)];
		}

		/*! \copydoc nemo::Network::getNeuronParameter */
		float getParameter(unsigned g_idx, unsigned param) const {
			return m_param[parameterIndex(param)][m_mapper.localIdx(g_idx)];
		}

		float getMembranePotential(unsigned g_idx) const {
			return getState(g_idx, m_type.membranePotential());
		}

		/*! \copydoc nemo::Network::setNeuron */
		void set(unsigned g_idx, unsigned nargs, const float args[]);

		/*! \copydoc nemo::Network::setNeuronState */
		void setState(unsigned g_idx, unsigned var, float val) {
			m_state[0][stateIndex(var)][m_mapper.localIdx(g_idx)] = val;
		}

		/*! \copydoc nemo::Network::setNeuronParameter */
		void setParameter(unsigned g_idx, unsigned param, float val) {
			m_param[parameterIndex(param)][m_mapper.localIdx(g_idx)] = val;
		}

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus
		 *
		 * \pre the internal firing stimulus buffer (\a m_fstim) is all false
		 */
		void setFiringStimulus(const std::vector<unsigned>& fstim);

		typedef RandomMapper<nidx_t> mapper_type;

		const mapper_type& mapper() const { return m_mapper; }

		/*! \return number of neurons in this collection */
		size_t size() const { return m_size; }

	private :

		mapper_type m_mapper;

		/*! Common type for all neurons in this collection */
		NeuronType m_type;

		const unsigned m_nParam;
		const unsigned m_nState;

		/*! Neurons are stored in several structure-of-arrays, supporting
		 * arbitrary neuron types. Functions modifying these maintain the
		 * invariant that the shapes are the same.
		 *
		 * The indices here are:
		 *
		 * 1. (outer) parameter index
		 * 2. (inner) neuron index
		 */
		typedef boost::multi_array<float, 2> param_type;
		param_type m_param;

		/*! Neuron state is stored in a structure-of-arrays format, supporting
		 * arbitrary neuron types. Functions modifying these maintain the
		 *
		 * The indices here are:
		 *
		 * 1. (outer) history index
		 * 2.         parameter index
		 * 3. (inner) neuron index
		 */
		typedef boost::multi_array<float, 3> state_type;
		state_type m_state;

		/*! Set neuron, like \a cpu::Neurons::set, but with a local index */
		void setLocal(unsigned l_idx, const float param[], const float state[]);

		/*! Number of neurons in this collection */
		size_t m_size;

		/*! \return parameter index after checking its validity */
		unsigned parameterIndex(unsigned i) const;

		/*! \return state variable index after checking its validity */
		unsigned stateIndex(unsigned i) const;

		/*! RNG with separate state for each neuron */
		std::vector<RNG> m_rng;

		/*! firing stimulus (for a single cycle).
		 *
		 * This is really a boolean vector, but use unsigned to support
		 * parallelisation
		 */
		std::vector<unsigned> m_fstim;

		//! \todo maintain firing buffer etc. here instead?

		/* The update function itself is found in a plugin which is loaded
		 * dynamically */
		Plugin m_plugin;
		cpu_update_neurons_t* m_update_neurons;

};


	}
}

#endif
