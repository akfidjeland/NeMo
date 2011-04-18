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

#include <nemo/RandomMapper.hpp>
#include <nemo/RNG.hpp>
#include <nemo/network/Generator.hpp>

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

		/*! Set up local storage for all neurons. Doing so also creates a
		 * mapping from global to dense local neuron indices. The resulting
		 * mapper can be queried via \a mapper */
		Neurons(const nemo::network::Generator& net);

		/*! Update the state of all neurons
		 *
		 * \post the input current vector is set to all zero.
		 * \post the internal firing stimulus buffer (\a m_fstim) is set to all false.
		 */
		void update(int start, int end, unsigned fbits,
			fix_t current[], uint64_t recentFiring[], unsigned fired[]);

		/*! \copydoc nemo::Network::getNeuronState */
		float getState(unsigned g_idx, unsigned var) const {
			return m_state[stateIndex(var)][m_mapper.localIdx(g_idx)];
		}

		/*! \copydoc nemo::Network::getNeuronParameter */
		float getParameter(unsigned g_idx, unsigned param) const {
			return m_param[parameterIndex(param)][m_mapper.localIdx(g_idx)];
		}

		float getMembranePotential(unsigned g_idx) const {
			return getState(g_idx, m_type.membranePotential());
		}

		/*! \copydoc nemo::Network::setNeuron */
		void set(unsigned g_idx, const float param[], const float state[]);

		/*! \copydoc nemo::Network::setNeuronState */
		void setState(unsigned g_idx, unsigned var, float val) {
			m_state[stateIndex(var)][m_mapper.localIdx(g_idx)] = val;
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

		/* Neurons are stored in several Structure-of-arrays, supporting
		 * arbitrary neuron types. Functions modifying these maintain the
		 * invariant that the shapes are the same. */
		std::vector< std::vector<float> > m_param;
		std::vector< std::vector<float> > m_state;

		/*! Number of neurons in this collection */
		size_t m_size;

		/*! \return array of parameter \a pidx
		 *
		 * The array contains the values for the given parameter for all the
		 * neurons in this collection.
		 */
		const float* parameterArray(unsigned pidx) const {
			return &(m_param.at(pidx)[0]);
		}

		/*! \return array of state variables \a sidx
		 *
		 * The array contains the values for the given parameter for all the
		 * neurons in this collection.
		 */
		float* stateArray(unsigned sidx) {
			return &(m_state.at(sidx)[0]);
		}

		/*! \return parameter index after checking its validity */
		unsigned parameterIndex(unsigned i) const;

		/*! \return state variable index after checking its validity */
		unsigned stateIndex(unsigned i) const;

		/*! Add a new neuron
		 *
		 * \param fParam array of floating point parameters
		 * \param fState array of floating point state variables
		 *
		 * \return local index (wihtin this class) of the newly added neuron
		 *
		 * \pre the input arrays have the lengths specified by the neuron type
		 * 		used when this object was created.
		 */
		size_t add(const float fParam[], const float fState[]);

		/*! RNG with separate state for each neuron */
		std::vector<nemo::RNG> m_rng;

		/*! firing stimulus (for a single cycle).
		 *
		 * This is really a boolean vector, but use unsigned to support
		 * parallelisation
		 */
		std::vector<unsigned> m_fstim;


		//! \todo maintain firing buffer etc. here instead?
};


	}
}

#endif
