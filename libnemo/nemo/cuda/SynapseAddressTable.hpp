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

#include <nemo/internal_types.h>

namespace nemo {

struct AddressRange
{
	unsigned start;
	unsigned end;

	AddressRange() : start(0), end(0) {}
	AddressRange(unsigned s, unsigned e) : start(s), end(e) {}

	bool valid() const { return end > start; }

	unsigned size() const { return end - start; }
};


/*! \brief Table specifying synapse addresses on the device 
 *
 * The connectivity data is stored in the format specified in \a ConnectivityMatrix.
 * Synapses for a particular neuron may be stored dispersed over several
 * non-contigous blocks of memory. If the user queries the connectivity data at
 * run-time, we need to be able to combine these blocks. This class stores the
 * address ranges for the relevant data on the device to facilitate this.
 * Additionally it stores the start warp and end warp for each neuron.
 */
class SynapseAddressTable
{
	public :

		// using default ctor

		void addBlock(nidx_t sourceNeuron, unsigned blockStart, unsigned blockEnd);

		void setWarpRange(nidx_t sourceNeuron, unsigned start, unsigned end);

		/*! \return start and end warps for the given neuron */
		const AddressRange& warpsOf(nidx_t sourceNeuron) const;

		const std::vector<AddressRange>& synapsesOf(nidx_t sourceNeuron) const;

	private : 

		// all the blocks for a single neuron
		typedef std::vector<AddressRange> neuron_ranges_t;

		// warps + blocks (word addresses) for a single neuron
		typedef std::pair<AddressRange, neuron_ranges_t> neuron_data_t;

		// all the blocks for the whole network 
		std::map<nidx_t, neuron_data_t > m_data;
};

} // end namespace nemo

#endif
