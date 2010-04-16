#ifndef NEMO_TYPES_H
#define NEMO_TYPES_H

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \todo have cmake check for presence of this file
#ifdef _WIN32
/* It seems that the 2010 version of Visual C++ have caught up to at least
 * *parts* of the C99 spec, in particular the stdint.h file. Versions prior to
 * this, however, do not include this file, so we have to use Alexander
 * Chemeris' version */
#include "win_stdint.h"
#else
#include <stdint.h>
#endif

typedef unsigned nidx_t; // neuron index
typedef unsigned sidx_t; // synapse index
typedef unsigned delay_t;
typedef float weight_t;  // on the host
typedef unsigned cycle_t;

#endif
