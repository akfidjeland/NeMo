#include "Network.hpp"
#include "RuntimeData.hpp"

namespace nemo {

Network*
Network::create(bool setReverse, unsigned maxReadPeriod)
{
	int dev = RuntimeData::selectDevice();
	return dev == -1 ? NULL : new RuntimeData(setReverse, maxReadPeriod);
}



Network::~Network()
{
	;
}

} // end namespace nemo
