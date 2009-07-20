#include <stddef.h>
#include <mex.h>

#include "error.h"


/* MEX's error function only accepts an already formatted string, so we need to
 * do our own formatting before calling it. */
void
error(char* fmt, ...)
{
	const size_t MAX_LEN = 512;
	char* mx_errorMsg = mxMalloc(MAX_LEN);
	va_list args;
    va_start(args, fmt );
	vsnprintf(mx_errorMsg, MAX_LEN, fmt, args);
	va_end(args);
	mexErrMsgTxt(mx_errorMsg);
}
