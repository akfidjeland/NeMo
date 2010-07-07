#ifndef CYCLE_CU
#define CYCLE_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \todo merge this with other files

#ifdef __DEVICE_EMULATION__

/* For logging we often need to print the current cycle
 * number. To avoid passing this around as an extra parameter
 * (which would either be conditionally compiled, or be a
 * source of unused parameter warnings), we just use a global
 * shared variable. */
__shared__ uint32_t s_cycle;

#endif

#endif
