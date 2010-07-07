#ifndef NEMO_MPI_LOG_HPP
#define NEMO_MPI_LOG_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/config.h>

#ifdef INCLUDE_MPI_LOGGING

#include <stdio.h>
#include <stdlib.h>

#define MPI_LOG(...) fprintf(stdout, __VA_ARGS__);

#else

#define MPI_LOG(...)

#endif

#endif
