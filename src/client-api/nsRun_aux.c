#include <mex.h>
#include <matrix.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "client.h"



/* Return the socket descriptor argument */
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



/* Return the number of simulation cycles to simulation */
unsigned int
getScalarInt(const mxArray* arr)
{
	if(!mxIsNumeric(arr)) {
		mxErrMsgTxt("NSTEPS argument should be numeric");
	}

	if(mxGetN(arr) != 1 || mxGetM(arr) != 1) {
		mexErrMsgTxt("NSTEPS argument should be scalar");
	}
	/* Generally, if the user specifies the argument in the MATLAB terminal it
	 * will be a double. However, the backend simulation is discrete, so
	 * fractional cycles make little sense. */
	double nstepsf;
	if(modf(mxGetScalar(arr), &nstepsf) != 0.0) {
		mxErrMsgTxt("NSTEPS argument should be integral");
	}

	unsigned int nsteps = (unsigned int) nstepsf;
	return nsteps;
}



/* Check that the array format is correct and return pointers to the two
 * columns */
void
getFiringStimulus(const mxArray* arr, 
		uint32_t** firingCycles,
		uint32_t** firingIdx,
		unsigned int* len)
{
	if(mxIsChar(arr) || mxIsClass(arr, "sparse") || mxIsComplex(arr) ) {
		mexErrMsgTxt("Firing stimulus matrix must be real, full, and nonstring");
	}

	if(!mxIsUint32(arr)) {
		mexErrMsgTxt("Firing stimulus matrix must have elements of type uint32");
	}

	if(mxGetN(arr) != 2 && mxGetM(arr) != 0) {
		mexErrMsgTxt("If non-empty, firing stimulus matrix must be of size (f,2)");
	}

	*len = mxGetM(arr);
	*firingCycles = (uint32_t*) mxGetData(arr);
	*firingIdx = *firingCycles + *len;
}



/*! Return annotated error while freeing up haskell-allocated error string */
void
error(const char* prefix, char* hs_errorMsg)
{
	size_t len = strlen(hs_errorMsg) + strlen(prefix) + 1;
	char* mx_errorMsg = mxMalloc(len);
	snprintf(mx_errorMsg, len, "%s%s", prefix, hs_errorMsg);
	free(hs_errorMsg);
	mexErrMsgTxt(mx_errorMsg);
}


/*! nsStep
 * 		run simulation for a fixed number of cycles and read back firing data.
 *
 * \param sockfd
 * 		Socket descriptor returned by earlier call to nsStart 
 * \param n
 * 		Number of cycles for which the simulation should run
 * \param firing_stimulus
 * 		(f,2) matrix where f is the number of forced firings during the next n
 * 		cycles. Each row specifies a cycle in the range [0, n) and a valid
 * 		neuron index. The rows should be sorted by cycle number. The \a
 * 		firing_stimulus matrix should contain unsigned integers (uint32).
 *
 * \return
 * 		(m,2) matrix containing time + firing index in each row, where m is the
 * 		total number of firings that took place in the most recent n cycles.
 */
void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(nrhs != 5) {
		mexErrMsgTxt("Incorrect number of arguments. Usage: nsStep(sockfd, nsteps, firing_stimulus, applySTDP, stdpReward)");
	}

	int32_t sockfd = getSockfd(prhs[0]); 
	unsigned int nsteps = getScalarInt(prhs[1]);

	uint32_t* firingStimulusCycles;
	uint32_t* firingStimulusIdx;
	unsigned int firingStimulusLen;
	getFiringStimulus(prhs[2], 
			&firingStimulusCycles,
			&firingStimulusIdx,
			&firingStimulusLen);

	uint32_t* firingCycles = NULL;
	uint32_t* firingIdx = NULL;
	unsigned int firingLen; 

	uint applySTDP = getScalarInt(prhs[3]);
	double stdpReward = mxGetScalar(prhs[4]);
	
	uint32_t elapsed = 0;      /* computation time on server in milliseconds */
	char* hs_errorMsg = NULL;  /* allocated on haskell side */
    bool success = hs_runSimulation(sockfd, nsteps,
                applySTDP, stdpReward,
				firingStimulusCycles, firingStimulusIdx, firingStimulusLen, 
				&firingCycles, &firingIdx, &firingLen, &elapsed, &hs_errorMsg);
	if(!success) {
		free(firingCycles);
		free(firingIdx);
		error("server error: ", hs_errorMsg);
	}

	if((plhs[0] = mxCreateNumericMatrix(firingLen, 2, mxUINT32_CLASS, mxREAL)) == NULL) { 
		free(firingCycles);
		free(firingIdx);
		mexErrMsgTxt("Failed to allocate memory for return data: firing vector\n");
	}

	uint32_t* ret = (uint32_t*) mxGetData(plhs[0]); 

	memcpy((char*)ret, firingCycles, firingLen*sizeof(uint32_t));
	memcpy((char*)(ret + firingLen), firingIdx, firingLen*sizeof(uint32_t));

	if((plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL)) == NULL) {
		mexErrMsgTxt("Failed to allocate memory for return data: elapsed time\n");
	}
	int32_t* retElapsed = (int32_t*) mxGetData(plhs[1]);
	retElapsed[0] = elapsed;

	free(firingCycles);
	free(firingIdx);
}
