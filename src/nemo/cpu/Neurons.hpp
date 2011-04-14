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

#include <nemo/Neurons.hpp>
#include <nemo/RandomMapper.hpp>
#include <nemo/RNG.hpp>
#include <nemo/network/Generator.hpp>

namespace nemo {
	namespace cpu {


/*! Neuron population for CPU backend
 *
 * The neurons are stored internally in dense structure-of-arrays. 
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
		 * The current vector is cleared 
		 */
		void update(int start, int end, unsigned fbits,
			unsigned fstim[], uint64_t recentFiring[],
			fix_t current[], unsigned fired[], nemo::RNG rng[]);

		/*! \copydoc nemo::Network::getNeuronState */
		float getState(unsigned g_idx, unsigned var) const {
			return m_neurons.getState(m_mapper.localIdx(g_idx), var);
		}

		/*! \copydoc nemo::Network::getNeuronParameter */
		float getParameter(unsigned g_idx, unsigned param) const {
			return m_neurons.getParameter(m_mapper.localIdx(g_idx), param);
		}

		float getMembranePotential(unsigned g_idx) const {
			return m_neurons.getMembranePotential(m_mapper.localIdx(g_idx));
		}

		/*! \copydoc nemo::Network::setNeuron */
		void set(unsigned g_idx, const float param[], const float state[]) {
			m_neurons.set(m_mapper.localIdx(g_idx), param, state);
		}

		/*! \copydoc nemo::Network::setNeuronState */
		void setState(unsigned g_idx, unsigned var, float val) {
			m_neurons.setState(m_mapper.localIdx(g_idx), var, val);
		}

		/*! \copydoc nemo::Network::setNeuronParameter */
		void setParameter(unsigned g_idx, unsigned var, float val) {
			m_neurons.setParameter(m_mapper.localIdx(g_idx), var, val);
		}

		typedef RandomMapper<nidx_t> mapper_type;

		const mapper_type& mapper() const { return m_mapper; }

	private :

		mapper_type m_mapper;

		//! \todo support multiple neuron types here
		nemo::Neurons m_neurons;

		//! \todo maintain firing buffer etc. here instead?
};


	}
}
#endif
