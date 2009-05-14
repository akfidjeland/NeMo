#ifndef TIME_HPP
#define TIME_HPP

#include <time.h>

double diffclock(clock_t clock1,clock_t clock2);

void printElapsed(const char* msg, clock_t begin);

#endif
