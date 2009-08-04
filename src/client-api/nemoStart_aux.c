#include <mex.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include "client.h"
#include "args.h"
#include "error.h"


#if defined(MINGW32) || defined(WIN32)
/* Setting the windows version is needed for correct linkage of getaddrinfo and
 * freeaddrinfo. This means that using the resulting dll will fail on versions
 * of Windows prior to XP. */
#	define WINVER 0x0501
#	include <winsock2.h>
#	include <ws2tcpip.h>
#else
#	include <string.h>
#	include <stdlib.h>
#	include <errno.h>
#	include <sys/types.h>
#	include <sys/socket.h>
#	include <netdb.h>
#	include <unistd.h>
#	include <stdarg.h>
#endif


void
connectionError(const char* call,
		const char* hostname,
		unsigned short portno,
		const char* errorMsg)
{
	mexPrintf("Error in '%s' %s:%u\n", call, hostname, portno);
	mexErrMsgTxt(errorMsg);
}


/*! \return socket file descriptor. Crash program on errors. */
int
connectSocket(const char* hostname, unsigned short portno)
{
	struct addrinfo hints;

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;      /* either IPv4 or IPv6 */
	hints.ai_socktype = SOCK_STREAM;  /* TCP stream sockets */

	int status;
	struct addrinfo* servinfo;
	char port[5]; /* ... for up to 2^16 ports, as per TCP spec */
	sprintf(port, "%u", portno);
	if((status = getaddrinfo(hostname, port, &hints, &servinfo)) == -1) {
		connectionError("getaddrinfo", hostname, portno, gai_strerror(status));
	}

	int sockfd;
	if((sockfd = socket(servinfo->ai_family, 
					servinfo->ai_socktype, 
					servinfo->ai_protocol)) == -1) {
		connectionError("socket", hostname, portno, strerror(errno));
	}

	if((status = connect(sockfd, servinfo->ai_addr, servinfo->ai_addrlen)) == -1) {
		/* errno may indicate "no error" here, at least when used through MEX */
		connectionError("connect", hostname, portno, strerror(errno));
	}

	freeaddrinfo(servinfo);
	return sockfd;
}



/* The haskell runtime system (RTS) has to be loaded before we can interact
 * with the client library. It is important that this is started only once and
 * shut down only once. Failure to do this causes matlab to segfault. 
 *
 * Under windows we also need to load and unload the winsock dll in the same
 * manner.
 *
 * Both of these are taken care of by calling startRTS. The shutdown is then
 * handled automatically once mex is all finished.
 */
void stopRTS();
void
startRTS()
{
	static bool clientStarted = false;
	if(!clientStarted) {
		Client_init();
#if defined(MINGW32) || defined(WIN32)
		WSADATA wsaData;   
		if (WSAStartup(MAKEWORD(1, 1), &wsaData) != 0) {
			mexErrMsgTxt("WSAStartup failed.");
		}
#endif
		mexAtExit(stopRTS);
		clientStarted = true;
	}
}


void
stopRTS()
{
	Client_exit();
#if defined(MINGW32) || defined(WIN32)
    WSACleanup();
#endif
}



void
setOrCheckLength(int found, int* expected,
		const char* dimension,
		const char* foundName,
		const char* expectedName)
{
	assert(expected != NULL);
	if(*expected == 0) {
		*expected = found;
	} else if(*expected != found) {
		const size_t MSGLEN = 256;
		char msg[MSGLEN];
		snprintf(msg, MSGLEN,
			"Arrays length mismatch for dimension %s: length(%s)=%d while length(%s)=%d\n",
			dimension,
			foundName, found,
			expectedName, expected);
		mexErrMsgTxt(msg);
	}
}


/*! \todo move into args.c */
/* If lenOut is NULL, ignore the argument. If it's 0, set it to the array
 * length. Otherwise, verify that the length of the array is the same */
double*
getDoubleArr(const mxArray* arr, int* lenOut,
		const char* arrname,
		const char* refarrname)
{
	checkRealFullNonstring(arr, arrname);
	if(mxGetN(arr) != 1) {
		error("Array '%s' should be (m,1). Actual dimensions are (%u,%u).",
				arrname, mxGetM(arr), mxGetN(arr));
	}
	setOrCheckLength(mxGetM(arr), lenOut, "M", arrname, refarrname);
	return mxGetPr(arr);
}


/*! \return (m,1) Array which might possibly be null */
double*
maybeGetDoubleArr(const mxArray* arr, int* len, const char* arrname)
{
	checkRealFullNonstring(arr, arrname);
	if(mxGetN(arr) != 1 && mxGetN(arr) != 0) {
		error("Array '%s' should be (m,1) or (m,0). Actual dimensions are (%u,%u).",
				arrname, mxGetM(arr), mxGetN(arr));
	}
	*len = mxGetM(arr);
	return mxGetPr(arr);

}


double*
getDoubleArr2(const mxArray* arr, int* stride, int* lenOut,
		const char* arrname,
		const char* refarrname)
{
	checkRealFullNonstring(arr, arrname);
	/* The matrix should be transposed before calling nsStart. (done in wrapper.) */
	setOrCheckLength(mxGetM(arr), stride, "M", arrname, refarrname);
	setOrCheckLength(mxGetN(arr), lenOut, "N", arrname, refarrname);
	return mxGetPr(arr);
}



int32_t*
getIntArr2(const mxArray* arr, int* stride, int* len,
		const char* arrname,
		const char* refarrname)
{
	if(mxIsChar(arr) || mxIsClass(arr, "sparse") || mxIsComplex(arr) ) {
		mexErrMsgTxt("Argument must be real, full, and nonstring");
	}
	if(!mxIsInt32(arr)) {
		mexErrMsgTxt("Integer array must have elements of type int32");
	}
	setOrCheckLength(mxGetM(arr), stride, "M", arrname, refarrname);
	setOrCheckLength(mxGetN(arr), len, "N", arrname, refarrname);
	return (int32_t*) mxGetData(arr);
}



void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	startRTS();

	if(nrhs != 16) {
		mexErrMsgTxt("Must have 18 arguments. Usage: nsStart(hostname, port, a, b, c, d, u, v, postIdx, delays, weights, temp_subres, stdp_active, stdp_prefire, stdp_postfire, maxWeight)");
	}

	if(nlhs != 1) {
		mexErrMsgTxt("Must have one output argument");
	}

	if(!mxIsChar(prhs[0])) {
		mexErrMsgTxt("First argument (hostname) should be a string");
	}

	int nlen = 0;
	double* a = getDoubleArr(prhs[2], &nlen, "a", "none");
	double* b = getDoubleArr(prhs[3], &nlen, "b", "a");
	double* c = getDoubleArr(prhs[4], &nlen, "c", "a");
	double* d = getDoubleArr(prhs[5], &nlen, "d", "a");
	double* u = getDoubleArr(prhs[6], &nlen, "u", "a");
	double* v = getDoubleArr(prhs[7], &nlen, "v", "a");


	int sstride = 0;
	int32_t* postIdx = getIntArr2(prhs[8], &sstride, &nlen, "postIdx", "none/a");
	int32_t* delays = getIntArr2(prhs[9], &sstride, &nlen, "delays", "postIdx");
	double* weights = getDoubleArr2(prhs[10], &sstride, &nlen, "weights", "postIdx");

    int tempSubres = getScalarInt(prhs[11], "temporal subresolution");

	char* hostname = mxArrayToString(prhs[0]);
	int port = getScalarInt(prhs[1], "port number");
	int sockfd = connectSocket(hostname, port ? port : hs_defaultPort());
	mxFree(hostname);

	int usingSTDP = getScalarInt(prhs[12], "using STDP");

	int prefire_len = 0;
	int postfire_len = 0;

	double* prefire  = maybeGetDoubleArr(prhs[13], &prefire_len, "prefire (STDP)");
	double* postfire = maybeGetDoubleArr(prhs[14], &postfire_len, "postfire (STDP)");
	double maxWeight = mxGetScalar(prhs[15]);

	bool success = hs_startSimulation(sockfd, nlen, sstride, tempSubres,
			usingSTDP, prefire_len, postfire_len, prefire, postfire, maxWeight,
			a, b, c, d, u, v, postIdx, delays, weights);
    if(!success) {
        mexErrMsgTxt("An unspecified error occurred on the host");
    }

	if((plhs[0] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL)) == NULL) {
		mexErrMsgTxt("Failed to allocate memory for return data\n");
	}

	int32_t* output = (int32_t*) mxGetData(plhs[0]);
	output[0] = sockfd;
}