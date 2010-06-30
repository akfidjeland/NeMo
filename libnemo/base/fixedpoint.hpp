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

/* Convert floating point to fixed-point */
fix_t fx_toFix(float f, unsigned fractionalBits);

/* Convert fixed-point to floating point */
float fx_toFloat(fix_t f, unsigned fractionalBits);

#endif
