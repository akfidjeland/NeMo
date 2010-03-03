#include "Network.hpp"
#include "RuntimeData.hpp"

namespace nemo {

Network*
Network::create(bool setReverse, unsigned maxReadPeriod)
{
	return new RuntimeData(setReverse, maxReadPeriod);
}

} // end namespace nemo
