#ifndef TIME_HPP
#define TIME_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <time.h>

double diffclock(clock_t clock1,clock_t clock2);

void printElapsed(const char* msg, clock_t begin);

#endif
