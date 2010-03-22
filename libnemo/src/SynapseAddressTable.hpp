#ifndef SYNAPSE_ADDRESS_TABLE
#define SYNAPSE_ADDRESS_TABLE

//! \file SynapseAddressTable.hpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>
#include <vector>

#include "nemo_types.h"

namespace nemo {


/*! \brief Table specifying synapse addresses on the device 
 *
 * The connectivity data is stored in the format specified in \a ConnectivityMatrix.
 * Synapses for a particular neuron may be stored dispersed over several
 * non-contigous blocks of memory. If the user queries the connectivity data at
 * run-time, we need to be able to combine these blocks. This class stores the
 * address ranges for the relevant data on the device to facilitate this.
 */
class SynapseAddressTable
{
	public :

		// using default ctor

		typedef std::pair<uint, uint> range_t; // a single contiguous block

		void addBlock(nidx_t sourceNeuron, uint blockStart, uint blockEnd);

		const std::vector<range_t>& synapsesOf(nidx_t sourceNeuron) const;

	private : 

		// all the blocks for a single neuron
		typedef std::vector<range_t> neuron_ranges_t;

		// all the blocks for the whole network 
		std::map<nidx_t, neuron_ranges_t> m_data;
};

} // end namespace nemo

#endif
