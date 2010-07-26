/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#define _GNU_SOURCE

#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sched.h>

#include "thread_affinity.h"

void
setThreadAffinity(unsigned core)
{
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core, &cpuset);
	pid_t tid = syscall(__NR_gettid);
	sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
}
