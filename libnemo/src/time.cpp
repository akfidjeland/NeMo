/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "time.hpp"
#include <stdio.h>

double 
diffclock(clock_t clock1,clock_t clock2)
{
    double diffticks=clock1-clock2;
    double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
    return diffms;
}


void
printElapsed(const char* msg, clock_t begin)
{
    clock_t end = clock();
    fprintf(stderr, "%s (%fms)\n", msg, diffclock(end, begin));
}

