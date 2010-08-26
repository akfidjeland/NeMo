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

#include <nemo/internal_types.h>
#include <nemo/config.h>


/* Convert floating point to fixed-point */
NEMO_BASE_DLL_PUBLIC
fix_t
fx_toFix(float f, unsigned fractionalBits);


/* Convert fixed-point to floating point */
NEMO_BASE_DLL_PUBLIC
float
fx_toFloat(fix_t v, unsigned fractionalBits);

#endif
