#include "CycleCounters.hpp"
#include "cycleCounting.cu_h"
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <fstream>


CycleCounters::CycleCounters(size_t partitionCount, int clockRateKHz, bool stdpEnabled) :
	m_ccMain(partitionCount, CC_MAIN_COUNT-1, true),
	m_ccApplySTDP(partitionCount, 1, stdpEnabled),
	m_partitionCount(partitionCount),
	m_clockRateKHz(clockRateKHz),
	m_stdpEnabled(stdpEnabled)
{ }


const char* durationNames[] = {
	"init",
	"random input",
	"load firing",
	"spike gather",
	"fire",
	"spike scatter",
	"L0 LTP",
	"L1 LTP",
	"update history"
};



void
printLine(
		const char* label,
		unsigned long long cycles,
		unsigned long long total,
		unsigned long long clockRateKHz,
		std::ofstream& outfile)
{
	unsigned long long timeMs = cycles / clockRateKHz;
	outfile << std::setw(15) << label << ":" 
		<< std::setw(10) << timeMs << "ms, "
		<< std::setw(15) << cycles << "cycles, "; 
	if(total != 0)
		outfile << std::setw(4) << 100*cycles/total << "%";
	outfile << std::endl;
}


void
CycleCounters::printCounterSet(
		NVector<unsigned long long>& cc_in,
		size_t counters,
		const char* setName,
		const char* names[], // for intermediate counters
		std::ofstream& outfile)
{
	const std::vector<unsigned long long>& cc = cc_in.copyFromDevice();
	//! \todo average over all partitions
	/* The data return by copyFromDevice is the raw device data, including any
	 * padding. Using cc.end() would therefore read too far */ 
	std::vector<unsigned long long>::const_iterator end = cc.begin() + counters;
	unsigned long long totalCycles = std::accumulate(cc.begin(), end, 0l);

	printLine(setName, totalCycles, totalCycles, m_clockRateKHz, outfile);
	outfile << std::endl;

	if(names != NULL) {
		for(std::vector<unsigned long long>::const_iterator i=cc.begin(); i != end; ++i) {
			unsigned long long cycles = *i;
			printLine(names[i-cc.begin()], cycles, totalCycles, m_clockRateKHz, outfile);
		}
		outfile << std::endl;
	}
}



void
CycleCounters::printCounters(const char* filename)
{
	std::ofstream outfile;
	outfile.open(filename);
	printCounterSet(m_ccMain, CC_MAIN_COUNT-1, "Main", durationNames, outfile);
	if(m_stdpEnabled) {
		printCounterSet(m_ccApplySTDP, 1, "STDP (apply)", NULL, outfile);
	}
	outfile.close();
}



unsigned long long*
CycleCounters::data() const
{
	//! \todo return data for different sets
	return m_ccMain.deviceData();
}


size_t
CycleCounters::pitch() const
{
	//! \todo return data for different sets
	return m_ccMain.wordPitch();
}
