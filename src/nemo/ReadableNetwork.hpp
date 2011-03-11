#ifndef NEMO_READABLE_NETWORK_HPP
#define NEMO_READABLE_NETWORK_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

namespace nemo {

/*! Network whose neurons and synapses can be queried
 *
 * Abstract base class
 */
class ReadableNetwork
{
	public :

		/*! \return source neuron id for a synapse */
		unsigned getSynapseSource(synapse_id id) const {
			return neuronIndex(id);
		}

		/*! \return target neuron id for a synapse */
		virtual unsigned getSynapseTarget(synapse_id) const = 0;

		/*! \return conductance delay for a synapse */
		virtual unsigned getSynapseDelay(synapse_id) const = 0;

		/*! \return weight for a synapse */
		virtual float getSynapseWeight(synapse_id) const = 0;

		/*! \return plasticity status for a synapse */
		virtual unsigned char getSynapsePlastic(synapse_id) const = 0;
};

}

#endif
