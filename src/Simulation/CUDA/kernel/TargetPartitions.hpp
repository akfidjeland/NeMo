#ifndef TARGET_PARTITIONS_HPP
#define TARGET_PARTITIONS_HPP

#include <map>
#include <set> 
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>

#include "targetPartitions.cu_h"

class TargetPartitions
{
	public :

		// default ctor is fine here

		void addTargetPartition(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				pidx_t targetPartition);

		void moveToDevice(size_t partitionCount, size_t partitionSize);

	private :

		boost::shared_ptr<targetp_t> md_arr; // device data
		size_t m_pitch;

		typedef boost::tuple<pidx_t, nidx_t> key_t;
		typedef std::set<targetp_t> row_t;
		typedef std::map<key_t, row_t> map_t;

		map_t m_acc;

		size_t maxPitch() const;
};

#endif
