#ifndef COMMON_H
#define COMMON_H

/* We don't actually know the line size. Current (2009) processors seem to use
 * 64B, so go with that. */
//! \todo perhaps use 256B. There should be no performance hit and it's more future-proof
#define ASSUMED_CACHE_LINE_SIZE 64

#endif
