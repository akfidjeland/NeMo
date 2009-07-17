#ifndef ARGS_H
#define ARGS_H

#include <mex.h>
#include <stdint.h>

/* Return the socket descriptor argument */
int32_t getSockfd(const mxArray* arr);

unsigned int getScalarInt(const mxArray* arr, const char* argname);

/* Checks that argument is real, full, and non-string, and returns with an
 * error if this is not the case */ 
void checkRealFullNonstring(const mxArray* arr, const char* argname);

#endif
