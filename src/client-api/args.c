#include <math.h>

#include "args.h"
#include "error.h"


int32_t
getSockfd(const mxArray* arr)
{
	if(!mxIsInt32(arr)) {
		mxErrMsgTxt("HANDLE should be int32_t");
	}

	if(mxGetN(arr) != 1 || mxGetM(arr) != 1) {
		mexErrMsgTxt("HANDLE should be scalar");
	}

	return *((int32_t*) mxGetData(arr));
}



unsigned int
getScalarInt(const mxArray* arr, const char* argname)
{
	if(!mxIsNumeric(arr)) {
		error("'%s' argument should be numeric", argname);
	}

	if(mxGetN(arr) != 1 || mxGetM(arr) != 1) {
		error("'%s' argument should be scalar", argname);

	}

	/* Generally, if the user specifies the argument in the MATLAB terminal it
	 * will be a double. We round these down */
	/*! \todo round to nearest? */
	double intpart;
	if(modf(mxGetScalar(arr), &intpart) != 0.0) {
		error("'%s' argument should be integral", argname);
	}

	return (unsigned int) intpart;
}



void
checkRealFullNonstring(const mxArray* arr, const char* argname)
{
	if(mxIsChar(arr) || mxIsClass(arr, "sparse") || mxIsComplex(arr) ) {
		error("Argument '%s' must be real, full, and nonstring", argname);
	}
}
