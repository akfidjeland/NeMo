#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <protocol/TBinaryProtocol.h>
#include <transport/TTransportUtils.h>
#include <transport/TSocket.h>

#include <boost/shared_ptr.hpp>

#include <mex.h>

#include <stdexcept>

#include <NemoFrontend.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#define CHECK_ARGS

using namespace boost;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

static shared_ptr<NemoFrontendClient> g_client;
static shared_ptr<TTransport> g_transport;



/* MEX's error function only accepts an already formatted string, so we need to
 * do our own formatting before calling it. */
void
error(const std::string& fmt, ...)
{
	/*! \todo use std::string instead for the message buffer */
	const size_t MAX_LEN = 512;
	char* mx_errorMsg = (char*) mxMalloc(MAX_LEN);
	va_list args;
    va_start(args, fmt);
	vsnprintf(mx_errorMsg, MAX_LEN, fmt.c_str(), args);
	va_end(args);
	mexErrMsgTxt(mx_errorMsg);
}


template<typename T>
T
scalar(const mxArray* arr)
{
#ifdef CHECK_ARGS
	if(mxGetN(arr) != 1 || mxGetM(arr) != 1) {
		error("argument should be scalar");
	}
#endif
	return *((T*) mxGetData(arr));
}



template<typename T>
std::vector<T> // let's hope the compiler can optimise the return...
vector(const mxArray* arr)
{
#ifdef CHECK_ARGS
	if(mxGetN(arr) != 1) {
		error("argument should be 1xm vector");
	}
#endif
	size_t length = mxGetM(arr);
	T* begin = mxGetPr(arr);
	return std::vector<T>(begin, begin+length);
}



const mxArray*
numeric(const mxArray* arr)
{
#ifdef CHECK_ARGS
	if(!mxIsNumeric(arr)) {
		error("argument should be numeric\n");
	}
#endif
	return arr;
}



/* Check that we're connected and issue an error if we're not */
void
checkConnection()
{
	if(g_transport == NULL || !g_transport->isOpen()) {
		error("Not connected to nemo\n"); 
	}
}



void
disconnect_()
{
	if(g_transport != NULL) {
		mexPrintf("disconnecting\n");
		g_transport->close();
	} else {
		mexPrintf("no active connection\n");
	}
}



void
disconnect(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	disconnect_();
}



void
connect(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// redirect thrift error messages
	GlobalOutput.setOutputFunction(mexWarnMsgTxt); 

	//! \todo get arguments from caller
	const std::string host = "localhost";
	int port = 56101;

	//! \todo what is this for?
	bool framed = false;

	shared_ptr<TSocket> socket(new TSocket(host, port));

	if (framed) {
		shared_ptr<TFramedTransport> framedSocket(new TFramedTransport(socket));
		g_transport = framedSocket;
	} else {
		shared_ptr<TBufferedTransport> bufferedSocket(new TBufferedTransport(socket));
		g_transport = bufferedSocket;
	}

	shared_ptr<TBinaryProtocol> protocol(new TBinaryProtocol(g_transport));
	shared_ptr<NemoFrontendClient> client(new NemoFrontendClient(protocol));
	g_client = client;

	try {
		g_transport->open();
		mexAtExit(disconnect_);
		mexPrintf("connect %s:%d\n", host.c_str(), port);
	} catch (TTransportException& ttx) {
		error("Connect failed: %s\n", ttx.what());
	} 

}



void
setNetwork(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkConnection();
	
	mexPrintf("sending network...");

	//! \todo check arguments in the wrapping matlab function
	int nlen = mxGetM(prhs[1]); // which should be the same as the others
	double* a = mxGetPr(prhs[1]);
	double* b = mxGetPr(prhs[2]);
	double* c = mxGetPr(prhs[3]);
	double* d = mxGetPr(prhs[4]);
	double* u = mxGetPr(prhs[5]);
	double* v = mxGetPr(prhs[6]);

	int sstride = mxGetM(prhs[7]);
	int32_t* targets = (int32_t*) mxGetData(prhs[7]);
	int32_t* delays = (int32_t*) mxGetData(prhs[8]);
	double* weights = mxGetPr(prhs[9]);

	std::map<int32_t, IzhNeuron> neurons;
	//! \todo perhaps just stick to 1-based addressing for a better interface
	for(int32_t n_idx=0; n_idx<nlen; ++n_idx) {

		IzhNeuron n;
		n.a = a[n_idx];
		n.b = b[n_idx];
		n.c = c[n_idx];
		n.d = d[n_idx];
		n.u = u[n_idx];
		n.v = v[n_idx];

		std::vector<Synapse> axon;

		for(size_t s_idx=n_idx*sstride; s_idx<(n_idx+1)*sstride; ++s_idx) {
			Synapse s;
			//! \todo remove source from axon in transport. It should be implicit
			s.source = n_idx;
			s.target = targets[s_idx];
			s.delay = delays[s_idx];
			s.weight = weights[s_idx];
			axon.push_back(s);
		}
	
		n.axon = axon;

		neurons.insert(std::make_pair(n_idx, n)); 
	}

	IzhNetwork network;
	network.neurons = neurons;
	g_client->setNetwork(network);

	mexPrintf("done\n");
}



/* Convert firing stimulus from Matlab matrix to wire format */
void
firingStimulus(unsigned ncycles,
		const mxArray* arr,
		std::vector<Stimulus>& stimuli)
{
	if(!mxIsUint32(arr)) {
		mexErrMsgTxt("Firing stimulus matrix must have elements of type uint32");
	}

	if(mxGetN(arr) != 2 && mxGetM(arr) != 0) {
		mexErrMsgTxt("If non-empty, firing stimulus matrix must be of size (f,2)");
	}

	size_t length = mxGetM(arr);
	uint32_t* cycles = (uint32_t*) mxGetData(arr);
	uint32_t* idx = cycles + length;
	uint32_t* end = idx + length;

	for(unsigned cycle=0; cycle < ncycles; ++cycle) {
		Stimulus stimulus; // for a single cycle
		while(cycles != end && *cycles == cycle) {
			stimulus.firing.push_back(*cycles);
			++cycles;
			++idx;
		}
		stimuli.push_back(stimulus);
	}
}



//! \todo factor this out into separate class to be re-used by other external clients
/* Convert firing data from wire format (\see nemo.thrift) to Matlab matrix
 * (m-by-2, one column for cycle number, one column for fired neuron index) */
mxArray*
firingData(const std::vector<Firing>& firing)
{
	size_t flen = 0;
	for(std::vector<Firing>::const_iterator i=firing.begin();
			i != firing.end(); ++i) {
		flen += i->size();
	}

	mxArray* ret = mxCreateNumericMatrix(flen, 2, mxUINT32_CLASS, mxREAL);

	uint32_t* cycles = (uint32_t*) mxGetData(ret);
	uint32_t* idx = cycles + flen;

	for(std::vector<Firing>::const_iterator i=firing.begin();
			i != firing.end(); ++i) {
		idx = std::copy(i->begin(), i->end(), idx);
		uint32_t cycle = i - firing.begin();
		size_t len = i->size();
		std::fill(cycles, cycles+len, cycle); 
		cycles += len;
	}

	return ret;
}



void
run(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	mexPrintf("run...");
	checkConnection();
	uint32_t ncycles = scalar<uint32_t>(numeric(prhs[1]));

	std::vector<Stimulus> stimuli;
	firingStimulus(ncycles, prhs[2], stimuli);

	std::vector<Firing> firing;
	g_client->run(firing, stimuli);

	plhs[0] = firingData(firing);
	mexPrintf("done\n");
}



void
enableStdp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkConnection();
	std::vector<double> prefire = vector<double>(numeric(prhs[1]));
	std::vector<double> postfire = vector<double>(numeric(prhs[2]));
	double maxWeight = scalar<double>(numeric(prhs[3]));
	g_client->enableStdp(prefire, postfire, maxWeight);
}



void
applyStdp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkConnection();
	double reward = scalar<double>(numeric(prhs[1]));
	g_client->applyStdp(reward);
}



void
disableStdp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkConnection();
	g_client->disableStdp();
}



#include <nemo_mex_fn_arr.hpp> // auto-generated table of function pointers

void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	/* The first argument specifies the function */
	uint32_t fn_idx = scalar<uint32_t>(numeric(prhs[0]));
	if(fn_idx >= FN_COUNT || fn_idx < 0) {
		error("nemo_mex: unknown function index: %u\n", fn_idx);
	}
	fn_arr[fn_idx](nlhs, plhs, nrhs, prhs);
}
