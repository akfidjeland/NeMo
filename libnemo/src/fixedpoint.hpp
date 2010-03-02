#ifndef FIXED_POINT_HPP
#define FIXED_POINT_HPP

#include "nemo_cuda_types.h"

/* Convert floating point to fixed-point */
fix_t fixedPoint(float f, unsigned fractionalBits);

void setFixedPointFormat(unsigned fractionalBits);

#endif
