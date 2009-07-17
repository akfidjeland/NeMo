#include <mex.h>
#include <matrix.h>
#include <string.h>
#include <stdint.h>

#include "error.h"
#include "args.h"



int32_t* targets;
uint32_t* delays;
double* weights; 


void
freeData(){
	if(targets != NULL) free(targets);
	if(delays != NULL) free(delays);
	if(weights != NULL) free(weights);

}


/* Return matrices */
enum {
	TARGETS = 0,
	DELAYS,
	WEIGHTS  
};


void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(nrhs != 1) {
		error("Function takes a single argument (socket fd)");
	}

	int32_t sockfd = getSockfd(prhs[0]);

	targets = NULL;
	delays = NULL;
	weights = NULL;

	uint ncount;
	uint scount;

	bool success = hs_getWeights(sockfd, &targets, &delays, &weights, &ncount, &scount);

	if(!success) {
		/*! \todo get error code from server */
		error("request for weights failed for unknown reason");
	}

	if((plhs[TARGETS] = mxCreateNumericMatrix(scount, ncount, mxINT32_CLASS, mxREAL)) == NULL) { 
		freeData();
		error("Failed to allocate memory for connectivity matrix (targets) return data\n");
	}

	if((plhs[DELAYS] = mxCreateNumericMatrix(scount, ncount, mxUINT32_CLASS, mxREAL)) == NULL) { 
		freeData();
		error("Failed to allocate memory for connectivity matrix (delays) return data\n");
	}

	if((plhs[WEIGHTS] = mxCreateNumericMatrix(scount, ncount, mxDOUBLE_CLASS, mxREAL)) == NULL) { 
		freeData();
		error("Failed to allocate memory for connectivity matrix (weights) return data\n");
	}
	
	/* The data returned from haskell should already be in the correct format,
	 * with invalid entries set to -1 (we need to adjust /all/ addresses */
	memcpy((char*) mxGetData(plhs[TARGETS]), targets, ncount*scount*sizeof(int32_t));
	memcpy((char*) mxGetData(plhs[DELAYS]),  delays,  ncount*scount*sizeof(uint32_t));
	memcpy((char*) mxGetData(plhs[WEIGHTS]), weights, ncount*scount*sizeof(double));

	freeData();
}
