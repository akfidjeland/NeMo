#include "CycleCounters.hpp"
#include "cycleCounting.cu_h"
#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <fstream>


CycleCounters::CycleCounters(size_t partitionCount, int clockRateKHz) :
	m_cc(partitionCount, DURATION_COUNT, true),
	m_partitionCount(partitionCount),
	m_clockRateKHz(clockRateKHz)
{ }


const char* durationNames[] = {
	"kernel",
	"init",
	"receive L1",
	"load firing",
	"integrate",
	"fire",
	"deliver L1",
	"store firing"
};


void
printLine(
		const char* label,
		unsigned long long cycles,
		unsigned long long total,
		unsigned long long clockRateKHz,
		std::ofstream& outfile)
{
	if(total == 0)
		return;
	unsigned long long timeMs = cycles / clockRateKHz;
	outfile << std::setw(15) << label << ":" 
		<< std::setw(10) << timeMs << "ms, "
		<< std::setw(15) << cycles << "cycles, " 
		<< std::setw(4) << 100*cycles/total << "%" << std::endl;
}


void
CycleCounters::printCounters(const char* filename)
{
	const std::vector<unsigned long long>& cc = m_cc.copyFromDevice();
	std::ofstream outfile;
	outfile.open(filename);
	unsigned long long totalCycles = cc.front();
	unsigned long long totalMeasured = 0;
	for(std::vector<unsigned long long>::const_iterator i=cc.begin();
			i != std::min(cc.begin()+DURATION_COUNT, cc.end()); ++i) {
		unsigned long long cycles = *i;
		printLine(durationNames[i-cc.begin()], cycles, 
				totalCycles, m_clockRateKHz, outfile);
		totalMeasured += cycles;
	}

	printLine("measured", totalMeasured-totalCycles, totalCycles, 
			m_clockRateKHz, outfile);
	outfile.close();
}



unsigned long long*
CycleCounters::data() const
{
	return m_cc.deviceData();
}



size_t
CycleCounters::pitch() const
{
	return m_cc.wordPitch();
}
