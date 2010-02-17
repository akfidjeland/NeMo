#ifndef CYCLE_CU
#define CYCLE_CU

#ifdef __DEVICE_EMULATION__

/* For logging we often need to print the current cycle number. To avoid
 * passing this around as an extra parameter (which would either be
 * conditionally compiled, or be a source of unused parameter warnings), we
 * just use a global shared variable. */
__shared__ uint32_t s_cycle;

#endif

#endif
