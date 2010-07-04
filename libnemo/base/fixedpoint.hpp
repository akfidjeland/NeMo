#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <types.h>
#include <nemo_config.h>


/* Convert floating point to fixed-point */
DLL_PUBLIC
fix_t
fx_toFix(float f, unsigned fractionalBits);


/* Convert fixed-point to floating point */
DLL_PUBLIC
float
fx_toFloat(fix_t v, unsigned fractionalBits);

#endif
