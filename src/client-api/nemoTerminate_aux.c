#include <mex.h>
#include <stdint.h>
#include "client.h"

#if defined(MINGW32) || defined(WIN32)
#	include <winsock2.h>
#endif

void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(nrhs != 1 || !mxIsInt32(prhs[0])) {
		mexErrMsgTxt("Must have a single 1x1 input containing socked descriptor (int32_t)");
	}

	if(nlhs != 0) {
		mexErrMsgTxt("Must have no outputs");
	}

	const int32_t sockfd = *((int32_t*) mxGetData(prhs[0]));
    if(!hs_stopSimulation(sockfd)) {
        /* We still need to make sure the socket is closed */
        mexWarnMsgTxt("nsTerminate_aux.c: an error occurred when trying to stop simulation");
    }
#if defined(MINGW32) || defined(WIN32)
    closesocket(sockfd);
#else
    close(sockfd);
#endif
}
