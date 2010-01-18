#ifndef OUTGOING_HPP
#define OUTGOING_HPP

#include <map>
#include <set> 
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>

#include "outgoing.cu_h"

class Outgoing
{
	public :

		// default ctor is fine here

		void addSynapse(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				pidx_t targetPartition);

		void moveToDevice(size_t partitionCount);

		outgoing_t* data() const { return md_arr.get(); }

		uint* count() const { return md_rowLength.get(); }

	private :

		boost::shared_ptr<outgoing_t> md_arr;  // device data
		size_t m_pitch;                       // max pitch

		boost::shared_ptr<uint> md_rowLength; // per-neuron pitch

		typedef boost::tuple<pidx_t, delay_t> tkey_t;
		typedef std::map<tkey_t, uint> targets_t;

		typedef boost::tuple<pidx_t, nidx_t> skey_t;
		typedef std::map<skey_t, targets_t> map_t;

		map_t m_acc;

		size_t maxPitch() const;
};

#endif
