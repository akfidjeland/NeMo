#include <vector>
#include <algorithm>
#include <string>
#ifdef _MSC_VER
#include <nemo/win_stdint.h> // in own local build directory
#else
#include <stdint.h>
#endif

#include <boost/numeric/conversion/cast.hpp>
#include <mex.h>


extern "C" {
#include <nemo.h>
}


/* We maintain pointers to all simulator objects as globals here, since Matlab
 * does not seem to have any support for foreign pointers.
 *
 * Instead of returning opaque poitners to the user we return indices into
 * these global vectors */ 

/* The user can specify multiple networks */
static std::vector<nemo_network_t> g_networks;

/* Additionally, the user can specify multiple configurations. */
static std::vector<nemo_configuration_t> g_configs;

//! \todo should we limit this to one?
static std::vector<nemo_simulation_t> g_sims;



/* When returning data to Matlab we need to specify the class of the data.
 * There should be a one-to-one mapping between the supported C types and
 * Matlab classes. The full list of supported classes can be found in 
 * /path-to-matlab/extern/include/matrix.h */
template<typename M> mxClassID classId() { 
	mexErrMsgIdAndTxt("nemo:api", "programming error: class id requested for unknown class");
	return mxUNKNOWN_CLASS;
}
template<> mxClassID classId<uint8_t>() { return mxUINT8_CLASS; }
template<> mxClassID classId<uint32_t>() { return mxUINT32_CLASS; }
template<> mxClassID classId<int32_t>() { return mxINT32_CLASS; }
template<> mxClassID classId<uint64_t>() { return mxUINT64_CLASS; }
template<> mxClassID classId<double>() { return mxDOUBLE_CLASS; }



void
checkNemoStatus(nemo_status_t s)
{
	if(s != NEMO_OK) {
		mexErrMsgIdAndTxt("nemo:backend", nemo_strerror());
	}
}



template<typename T>
const mxArray*
numeric(const mxArray* arr)
{
	if(!mxIsNumeric(arr)) {
		mexErrMsgIdAndTxt("nemo:api", "argument should be numeric\n");
	}
	if(mxGetClassID(arr) != classId<T>()) {
		/* to be able to report the expected class name we have to create a
		 * temporary array, as there's no simpler way of getting the name. 
		 * Matlab will garbage collect this. */
		mxArray* tmp = mxCreateNumericMatrix(1, 1, classId<T>(), mxREAL);
		mexErrMsgIdAndTxt("nemo:api", "expected input of class %s, but found %s",
				mxGetClassName(tmp), mxGetClassName(arr));
	}
	return arr;
}



/* Return scalar from a numeric 1x1 array provided by Matlab. M and N refers to
 * Matlab and Nemo types, rather than array dimensions. */
template<typename N, typename M>
N
scalar(const mxArray* arr)
{
	if(mxGetN(arr) != 1 || mxGetM(arr) != 1) {
		mexErrMsgIdAndTxt("nemo:api", "argument should be scalar");
	}
	// target, source
	return boost::numeric_cast<N, M>(*static_cast<M*>(mxGetData(numeric<M>(arr))));
}



/* Return vector from a numeric 1 x m array provided by Matlab. M and N refers
 * to Matlab and Nemo types, rather than array dimensions. */
template<typename N, typename M>
std::vector<N> // let's hope the compiler can optimise the return...
vector(const mxArray* arr)
{
	if(mxGetM(arr) != 1) {
		mexErrMsgIdAndTxt("nemo:api", 
				"argument should be 1 x m vector. Size is %u x %u",
				mxGetM(arr), mxGetN(arr));
	}

	size_t length = mxGetN(arr);
	std::vector<N> ret;
	M* begin = static_cast<M*>(mxGetData(numeric<M>(arr)));
	std::transform(begin, begin + length,
			std::back_inserter(ret),
			boost::numeric_cast<N, M>);
	return ret;
}



void
checkInputCount(int actualArgs, int expectedArgs)
{
	// The function id and handle are always an extra parameter
	if(actualArgs - 2 != expectedArgs) {
		mexErrMsgIdAndTxt("nemo:mex", "found %u input arguments, but expected %u",
			actualArgs - 2, expectedArgs);
	}
}


void
checkOutputCount(int actualArgs, int expectedArgs)
{
	if(actualArgs != expectedArgs) {
		mexErrMsgIdAndTxt("nemo:mex", "found %u output arguments, but expected %u",
			actualArgs, expectedArgs);
	}
}


/* Return numeric scalar in output argument 0 after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnScalar(mxArray* plhs[], int argno, const N& val)
{
	plhs[argno] = mxCreateNumericMatrix(1, 1, classId<M>(), mxREAL);
	*(static_cast<M*>(mxGetData(plhs[argno]))) = val;
}



/* Return numeric vector in output argument n after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnVector(mxArray* plhs[], int argno, const std::vector<N>& vec)
{
	mxArray* ret = mxCreateNumericMatrix(1, vec.size(), classId<M>(), mxREAL);
	std::transform(vec.begin(), vec.end(), 
			static_cast<M*>(mxGetData(ret)),
			boost::numeric_cast<N, M>);
	plhs[argno] = ret;
}


/* Return numeric vector in output argument n after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnVector(mxArray* plhs[], int argno, N* arr, unsigned len)
{
	mxArray* ret = mxCreateNumericMatrix(1, len, classId<M>(), mxREAL);
	std::transform(arr, arr+len,
			static_cast<M*>(mxGetData(ret)),
			boost::numeric_cast<N, M>);
	plhs[argno] = ret;
}



//! \todo add collection name?
//put collection inside struct instead
uint32_t
getHandleId(std::vector<void*> collection, const mxArray* prhs[], unsigned argno)
{
	uint32_t id = scalar<uint32_t, uint32_t>(prhs[argno]);
	if(id >= collection.size()) {
		mexErrMsgIdAndTxt("nemo:mex", "handle id %u out of bounds (%u items)",
				id, collection.size());
	}
	return id;
}



void*
getHandle(std::vector<void*>& collection, const mxArray* prhs[], unsigned argno)
{
	uint32_t id = getHandleId(collection, prhs, argno);
	void* ptr = collection.at(id);
	if(ptr == NULL) {
		//! \todo add collection name
		mexErrMsgIdAndTxt("nemo:mex", "handle id %u is NULL", id);
	}
	return ptr;
}



nemo_network_t
getNetwork(const mxArray* prhs[], unsigned argno)
{
	return getHandle(g_networks, prhs, argno);
}



nemo_configuration_t
getConfiguration(const mxArray* prhs[], unsigned argno)
{
	return getHandle(g_configs, prhs, argno);
}



nemo_simulation_t
getSimulation(const mxArray* prhs[], unsigned argno)
{
	return getHandle(g_sims, prhs, argno);
}



void
newNetwork(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	uint32_t id = g_networks.size();
	nemo_network_t net = nemo_new_network();
	g_networks.push_back(net);
	returnScalar<uint32_t, uint32_t>(plhs, 0, id);
}



void
deleteNetwork(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	uint32_t id = getHandleId(g_networks, prhs, 1);
	void* ptr = g_networks.at(id);
	if(ptr != NULL) {
		nemo_delete_network(ptr);
		g_networks.at(id) = NULL;
	}
}



void
newConfiguration(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	uint32_t id = g_configs.size();
	nemo_configuration_t conf = nemo_new_configuration();
	g_configs.push_back(conf);
	returnScalar<uint32_t, uint32_t>(plhs, 0, id);
}




void
deleteConfiguration(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	uint32_t id = getHandleId(g_configs, prhs, 1);
	void* ptr = g_configs.at(id);
	if(ptr != NULL) {
		nemo_delete_configuration(ptr);
		g_configs.at(id) = NULL;
	}
}



void
newSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	uint32_t id = g_sims.size(); 

	nemo_network_t net = getNetwork(prhs, 1);
	nemo_configuration_t conf = getConfiguration(prhs, 2);
	nemo_simulation_t sim = nemo_new_simulation(net, conf);
	if(sim == NULL) {
		mexErrMsgIdAndTxt("nemo:backend", "failed to create simulation: %s", nemo_strerror());
	}
	g_sims.push_back(sim);	
	returnScalar<uint32_t, uint32_t>(plhs, 0, id);
}



void
deleteSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	uint32_t id = getHandleId(g_sims, prhs, 1);
	void* ptr = g_sims.at(id);
	if(ptr != NULL) {
		nemo_delete_simulation(ptr);
		g_sims.at(id) = NULL;
	}
}



void
readFiring(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	unsigned* cycles;
	unsigned* nidx;
	unsigned nfired;
	unsigned ncycles;
	checkNemoStatus(nemo_read_firing(getSimulation(prhs, 1), &cycles, &nidx, &nfired, &ncycles));
	returnVector<unsigned, uint32_t>(plhs, 0, cycles, nfired);
	returnVector<unsigned, uint32_t>(plhs, 1, nidx, nfired);
}




void
addSynapses(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 5);
	nemo_network_t net = getNetwork(prhs, 1);
	std::vector<unsigned> targets = vector<unsigned, uint32_t>(prhs[3]);
	std::vector<unsigned> delays = vector<unsigned, uint32_t>(prhs[4]);
	std::vector<float> weights = vector<float, double>(prhs[5]);
	std::vector<unsigned char> plastic = vector<unsigned char, uint8_t>(prhs[6]);
	nemo_status_t ret = nemo_add_synapses(net,
			scalar<unsigned, uint32_t>(prhs[2]),
			&targets[0],
			&delays[0],
			&weights[0],
			&plastic[0],
			//! \todo check that all inputs are the same length
			mxGetN(prhs[3]));
	checkNemoStatus(ret);
}



void
getSynapses(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 1);

	unsigned* targets;
	unsigned* delays;
	float* weights;
	unsigned char* plastic;
	size_t len;

	checkNemoStatus(
			nemo_get_synapses(
					getSimulation(prhs, 1),
					scalar<unsigned, uint32_t>(prhs[2]),
					&targets,
					&delays,
					&weights,
					&plastic,
					&len
			)
	);

	returnVector<unsigned, uint32_t>(plhs, 0, targets, len);
	returnVector<unsigned, uint32_t>(plhs, 1, delays, len);
	returnVector<float, double>(plhs, 2, weights, len);
	returnVector<unsigned char, uint8_t>(plhs, 3, plastic, len);
}
/* AUTO-GENERATED CODE START */

void
addNeuron(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 8);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_add_neuron( 
                    getNetwork(prhs, 1), 
                    scalar<unsigned,uint32_t>(prhs[2]), 
                    scalar<float,double>(prhs[3]), 
                    scalar<float,double>(prhs[4]), 
                    scalar<float,double>(prhs[5]), 
                    scalar<float,double>(prhs[6]), 
                    scalar<float,double>(prhs[7]), 
                    scalar<float,double>(prhs[8]), 
                    scalar<float,double>(prhs[9]) 
            ) 
    );
}


void
addSynapse(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 5);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_add_synapse( 
                    getNetwork(prhs, 1), 
                    scalar<unsigned,uint32_t>(prhs[2]), 
                    scalar<unsigned,uint32_t>(prhs[3]), 
                    scalar<unsigned,uint32_t>(prhs[4]), 
                    scalar<float,double>(prhs[5]), 
                    scalar<unsigned char,uint8_t>(prhs[6]) 
            ) 
    );
}


void
neuronCount(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned ncount;
    checkNemoStatus(nemo_neuron_count(getNetwork(prhs, 1), &ncount));
    returnScalar<unsigned, uint32_t>(plhs, 0, ncount);
}


void
step(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    std::vector<unsigned> fstim = vector<unsigned, uint32_t>(prhs[2]);
    checkNemoStatus(nemo_step(getSimulation(prhs, 1), &fstim[0], fstim.size()));
}


void
applyStdp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_apply_stdp(getSimulation(prhs, 1), scalar<float,double>(prhs[2])) 
    );
}


void
flushFiringBuffer(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 0);
    checkNemoStatus(nemo_flush_firing_buffer(getSimulation(prhs, 1)));
}


void
elapsedWallclock(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned long elapsed;
    checkNemoStatus(nemo_elapsed_wallclock(getSimulation(prhs, 1), &elapsed));
    returnScalar<unsigned long, uint64_t>(plhs, 0, elapsed);
}


void
elapsedSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned long elapsed;
    checkNemoStatus(nemo_elapsed_simulation(getSimulation(prhs, 1), &elapsed));
    returnScalar<unsigned long, uint64_t>(plhs, 0, elapsed);
}


void
resetTimer(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 0);
    checkNemoStatus(nemo_reset_timer(getSimulation(prhs, 1)));
}


void
setCpuBackend(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_set_cpu_backend(getConfiguration(prhs, 1), scalar<int,int32_t>(prhs[2])) 
    );
}


void
setCudaBackend(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_set_cuda_backend(getConfiguration(prhs, 1), scalar<int,int32_t>(prhs[2])) 
    );
}


void
setCudaFiringBufferLength(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_set_cuda_firing_buffer_length( 
                    getConfiguration(prhs, 1), 
                    scalar<unsigned,uint32_t>(prhs[2]) 
            ) 
    );
}


void
cudaFiringBufferLength(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned milliseconds;
    checkNemoStatus( 
            nemo_cuda_firing_buffer_length(getConfiguration(prhs, 1), &milliseconds) 
    );
    returnScalar<unsigned, uint32_t>(plhs, 0, milliseconds);
}


void
setStdpFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 4);
    checkOutputCount(nlhs, 0);
    std::vector<float> prefire = vector<float, double>(prhs[2]);
    std::vector<float> postfire = vector<float, double>(prhs[3]);
    checkNemoStatus( 
            nemo_set_stdp_function( 
                    getConfiguration(prhs, 1), 
                    &prefire[0], prefire.size(), 
                    &postfire[0], postfire.size(), 
                    scalar<float,double>(prhs[4]), 
                    scalar<float,double>(prhs[5]) 
            ) 
    );
}


void
setFractionalBits(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_set_fractional_bits( 
                    getConfiguration(prhs, 1), 
                    scalar<unsigned,uint32_t>(prhs[2]) 
            ) 
    );
}


typedef void (*fn_ptr)(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

#define FN_COUNT 24

fn_ptr fn_arr[FN_COUNT] = {
	newNetwork,
	deleteNetwork,
	addNeuron,
	addSynapse,
	addSynapses,
	neuronCount,
	newSimulation,
	deleteSimulation,
	step,
	applyStdp,
	readFiring,
	flushFiringBuffer,
	getSynapses,
	elapsedWallclock,
	elapsedSimulation,
	resetTimer,
	newConfiguration,
	deleteConfiguration,
	setCpuBackend,
	setCudaBackend,
	setCudaFiringBufferLength,
	cudaFiringBufferLength,
	setStdpFunction,
	setFractionalBits};

/* AUTO-GENERATED CODE END */


void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	/* The first argument specifies the function */
	uint32_t fn_idx = scalar<uint32_t, uint32_t>(prhs[0]);
	if(fn_idx >= FN_COUNT || fn_idx < 0) {
		mexErrMsgIdAndTxt("nemo:mex", "Unknown function index %u", fn_idx);
	}
	fn_arr[fn_idx](nlhs, plhs, nrhs, prhs);
}
