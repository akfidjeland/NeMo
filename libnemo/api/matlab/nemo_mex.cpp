#include <vector>
#include <algorithm>
#include <string>

#include <boost/numeric/conversion/cast.hpp>
#include <mex.h>

#include <nemo.h>
#include <nemo/types.h>


/* Unique global objects containing all NeMo state */
static nemo_network_t g_network = NULL;
static nemo_configuration_t g_configuration = NULL;
static nemo_simulation_t g_simulation = NULL;



/* When returning data to Matlab we need to specify the class of the data.
 * There should be a one-to-one mapping between the supported C types and
 * Matlab classes. The full list of supported classes can be found in 
 * /path-to-matlab/extern/include/matrix.h */
template<typename M> mxClassID classId() { 
	mexErrMsgIdAndTxt("nemo:mex", "programming error: class id requested for unknown class");
	return mxUNKNOWN_CLASS;
}
template<> mxClassID classId<char*>() { return mxCHAR_CLASS; }
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



/* Return scalar from a given position in a Matlab array. M and N refers to
 * Matlab and Nemo types, rather than array dimensions.  */
template<typename N, typename M>
N
scalarAt(const mxArray* arr, size_t offset)
{
	/* No bounds checking here since we have already verified the bonds for all input vectors */
	return boost::numeric_cast<N, M>(*(static_cast<M*>(mxGetData(numeric<M>(arr)))+offset));
}



/* Return vector from a numeric 1 x m array provided by Matlab. M and N refers
 * to Matlab and Nemo types, rather than array dimensions. */
template<typename N, typename M>
std::vector<N> // let's hope the compiler can optimise the return...
vector(const mxArray* arr)
{
	std::vector<N> ret;
	if(mxGetM(arr) == 0 && mxGetN(arr) == 0) {
		return ret;
	}

	if(mxGetM(arr) != 1) {
		mexErrMsgIdAndTxt("nemo:api", 
				"argument should be 1 x m vector. Size is %u x %u",
				mxGetM(arr), mxGetN(arr));
	}

	size_t length = mxGetN(arr);
	M* begin = static_cast<M*>(mxGetData(numeric<M>(arr)));
	std::transform(begin, begin + length,
			std::back_inserter(ret),
			boost::numeric_cast<N, M>);
	return ret;
}



void
checkInputCount(int actualArgs, int expectedArgs)
{
	// The function id is always an extra parameter
	if(actualArgs - 1 != expectedArgs) {
		mexErrMsgIdAndTxt("nemo:api", "found %u input arguments, but expected %u",
			actualArgs - 1, expectedArgs);
	}
}


void
checkOutputCount(int actualArgs, int expectedArgs)
{
	if(actualArgs != expectedArgs) {
		mexErrMsgIdAndTxt("nemo:api", "found %u output arguments, but expected %u",
			actualArgs, expectedArgs);
	}
}



/* Print out all dimensions and lengths of all arguments (either input or output) */
void
reportVectorDimensions(int argc, const mxArray* argv[])
{
	for(int i=0; i < argc; ++i) {
		size_t n = mxGetN(argv[i]);
		size_t m = mxGetM(argv[i]);
		mexPrintf("argument %u: size=(%u,%u) length=%u\n", i, m, n, m*n);
	}
}



/* For the vector form of functions which are scalar in the C++ API (i.e. all
 * inputs and outputs are scalars) we allow using a vector form in Matlab, but
 * require that all vectors have the same dimensions. Return this and report
 * error if sizes are not the same. The precise shape of the matrices do not
 * matter. */
size_t
vectorDimension(int nrhs, const mxArray* prhs[])
{
	if(nrhs < 1) {
		mexErrMsgIdAndTxt("nemo:api", "function should have at least one input argument");
	}
	size_t dim = mxGetN(prhs[0]) * mxGetM(prhs[0]);
	for(int i=1; i < nrhs; ++i) {
		size_t found = mxGetN(prhs[i]) * mxGetM(prhs[i]);
		if(found != dim) {
			reportVectorDimensions(nrhs, prhs);
			mexErrMsgIdAndTxt("nemo:api", "vector arguments do not have the same size");
		}
	}
	return dim;
}



/* Return numeric scalar in output argument 0 after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnScalar(mxArray* plhs[], int argno, N val)
{
	plhs[argno] = mxCreateNumericMatrix(1, 1, classId<M>(), mxREAL);
	*(static_cast<M*>(mxGetData(plhs[argno]))) = val;
}


template<>
void
returnScalar<const char*, char*>(mxArray* plhs[], int argno, const char* str)
{
	plhs[argno] = mxCreateString(str);
}



/* Allocate memory for return vector */
template<typename M>
void
allocateOutputVector(mxArray* plhs[], int argno, size_t len)
{
	plhs[argno] = mxCreateNumericMatrix(1, len, classId<M>(), mxREAL);
}



/* Set an element in output array and do the required casting from the type
 * used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnScalarAt(mxArray* plhs[], int argno, size_t offset, N val)
{
	static_cast<M*>(mxGetData(plhs[argno]))[offset] = boost::numeric_cast<N, M>(val);
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



void
deleteGlobals()
{
	if(g_simulation != NULL) {
		nemo_delete_simulation(g_simulation);
		g_simulation = NULL;
	}
	if(g_network != NULL) {
		nemo_delete_network(g_network);
		g_network = NULL;
	}
	if(g_configuration != NULL) {
		nemo_delete_configuration(g_configuration);
		g_configuration = NULL;
	}
}



nemo_network_t
getNetwork()
{
	if(g_network == NULL) {
		g_network = nemo_new_network();
		mexAtExit(deleteGlobals);
	}
	return g_network;
}



void
clearNetwork(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(g_network == NULL) {
		/* This should not be considered a usage an error. If the user has not
		 * added any neurons or synapses the network object is NULL. Clearing
		 * this should be perfectly valid. */
		return;
	}
	nemo_delete_network(g_network);
	g_network = NULL;
}



nemo_configuration_t
getConfiguration()
{
	if(g_configuration == NULL) {
		g_configuration = nemo_new_configuration();
		mexAtExit(deleteGlobals);
	}
	return g_configuration;
}



/*! Reset the configuration object to the default settings */
void
resetConfiguration(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//! \todo do a swap here instead, to avoid dangling pointers 
	if(g_configuration != NULL) {
		nemo_delete_configuration(g_configuration);
	}
	g_configuration = nemo_new_configuration();
}


/*! The simulation only exists between calls to \a createSimulation and
 * \a destroySimulation. Asking for the simulation object outside this region is
 * an error. */
nemo_simulation_t
getSimulation()
{
	if(g_simulation == NULL) {
		//! \todo add a more helpful error message
		mexErrMsgIdAndTxt("nemo:api", "non-existing simulation object requested");
	}
	return g_simulation;
}



void
createSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	nemo_network_t net = getNetwork();
	nemo_configuration_t conf = getConfiguration();
	nemo_simulation_t sim = nemo_new_simulation(net, conf);
	if(sim == NULL) {
		mexErrMsgIdAndTxt("nemo:backend", "failed to create simulation: %s", nemo_strerror());
	}
	g_simulation = sim;
	mexAtExit(deleteGlobals);
}



void
destroySimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(g_simulation == NULL) {
		mexErrMsgIdAndTxt("nemo:api", "Attempt to stop simulation when simulation is not running");
	}
	//! \todo do a swap for exception safety?
	nemo_delete_simulation(g_simulation);
	g_simulation = NULL;
}
/* AUTO-GENERATED CODE START */

void
addNeuron(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    size_t elems = vectorDimension(8, prhs + 1);
    checkInputCount(nrhs, 8);
    checkOutputCount(nlhs, 0);
    void* hdl = getNetwork();
    for(size_t i=0; i<elems; ++i){
        checkNemoStatus( 
                nemo_add_neuron( 
                        hdl, 
                        scalarAt<unsigned,uint32_t>(prhs[1], i), 
                        scalarAt<float,double>(prhs[2], i), 
                        scalarAt<float,double>(prhs[3], i), 
                        scalarAt<float,double>(prhs[4], i), 
                        scalarAt<float,double>(prhs[5], i), 
                        scalarAt<float,double>(prhs[6], i), 
                        scalarAt<float,double>(prhs[7], i), 
                        scalarAt<float,double>(prhs[8], i) 
                ) 
        );
    }
}



void
addSynapse(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    size_t elems = vectorDimension(5, prhs + 1);
    checkInputCount(nrhs, 5);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<uint64_t>(plhs, 0, elems);
    void* hdl = getNetwork();
    for(size_t i=0; i<elems; ++i){
        uint64_t id;
        checkNemoStatus( 
                nemo_add_synapse( 
                        hdl, 
                        scalarAt<unsigned,uint32_t>(prhs[1], i), 
                        scalarAt<unsigned,uint32_t>(prhs[2], i), 
                        scalarAt<unsigned,uint32_t>(prhs[3], i), 
                        scalarAt<float,double>(prhs[4], i), 
                        scalarAt<unsigned char,uint8_t>(prhs[5], i), 
                        &id 
                ) 
        );
        returnScalarAt<uint64_t, uint64_t>(plhs, 0, i, id);
    }
}



void
neuronCount(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned ncount;
    checkNemoStatus(nemo_neuron_count(getNetwork(), &ncount));
    returnScalar<unsigned, uint32_t>(plhs, 0, ncount);
}



void
setCpuBackend(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_set_cpu_backend(getConfiguration(), scalar<int,int32_t>(prhs[1])) 
    );
}



void
setCudaBackend(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_set_cuda_backend(getConfiguration(), scalar<int,int32_t>(prhs[1])) 
    );
}



void
setStdpFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 4);
    checkOutputCount(nlhs, 0);
    std::vector<float> prefire = vector<float, double>(prhs[1]);
    std::vector<float> postfire = vector<float, double>(prhs[2]);
    checkNemoStatus( 
            nemo_set_stdp_function( 
                    getConfiguration(), 
                    &prefire[0], prefire.size(), 
                    &postfire[0], postfire.size(), 
                    scalar<float,double>(prhs[3]), 
                    scalar<float,double>(prhs[4]) 
            ) 
    );
}



void
backendDescription(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    const char* description;
    checkNemoStatus(nemo_backend_description(getConfiguration(), &description));
    returnScalar<const char*, char*>(plhs, 0, description);
}



void
step(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 3);
    checkOutputCount(nlhs, 1);
    std::vector<unsigned> fstim = vector<unsigned, uint32_t>(prhs[1]);
    std::vector<unsigned> istim_nidx = vector<unsigned, uint32_t>(prhs[2]);
    std::vector<float> istim_current = vector<float, double>(prhs[3]);
    unsigned* fired;
    size_t fired_len;
    checkNemoStatus( 
            nemo_step( 
                    getSimulation(), 
                    &fstim[0], fstim.size(), 
                    &istim_nidx[0], 
                    &istim_current[0], istim_current.size(), 
                    &fired, &fired_len 
            ) 
    );
    returnVector<unsigned, uint32_t>(plhs, 0, fired, fired_len);
}



void
applyStdp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_apply_stdp(getSimulation(), scalar<float,double>(prhs[1])) 
    );
}



void
getTargets(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    std::vector<uint64_t> synapses = vector<uint64_t, uint64_t>(prhs[1]);
    unsigned* targets;
    checkNemoStatus( 
            nemo_get_targets(getSimulation(), &synapses[0], synapses.size(), &targets) 
    );
    returnVector<unsigned, uint32_t>(plhs, 0, targets, synapses.size());
}



void
getDelays(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    std::vector<uint64_t> synapses = vector<uint64_t, uint64_t>(prhs[1]);
    unsigned* delays;
    checkNemoStatus( 
            nemo_get_delays(getSimulation(), &synapses[0], synapses.size(), &delays) 
    );
    returnVector<unsigned, uint32_t>(plhs, 0, delays, synapses.size());
}



void
getWeights(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    std::vector<uint64_t> synapses = vector<uint64_t, uint64_t>(prhs[1]);
    float* weights;
    checkNemoStatus( 
            nemo_get_weights(getSimulation(), &synapses[0], synapses.size(), &weights) 
    );
    returnVector<float, double>(plhs, 0, weights, synapses.size());
}



void
getPlastic(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    std::vector<uint64_t> synapses = vector<uint64_t, uint64_t>(prhs[1]);
    unsigned char* plastic;
    checkNemoStatus( 
            nemo_get_plastic(getSimulation(), &synapses[0], synapses.size(), &plastic) 
    );
    returnVector<unsigned char, uint8_t>(plhs, 0, plastic, synapses.size());
}



void
elapsedWallclock(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned long elapsed;
    checkNemoStatus(nemo_elapsed_wallclock(getSimulation(), &elapsed));
    returnScalar<unsigned long, uint64_t>(plhs, 0, elapsed);
}



void
elapsedSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned long elapsed;
    checkNemoStatus(nemo_elapsed_simulation(getSimulation(), &elapsed));
    returnScalar<unsigned long, uint64_t>(plhs, 0, elapsed);
}



void
resetTimer(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 0);
    checkNemoStatus(nemo_reset_timer(getSimulation()));
}



typedef void (*fn_ptr)(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
#define FN_COUNT 20
fn_ptr fn_arr[FN_COUNT] = {
    addNeuron,
    addSynapse,
    neuronCount,
    clearNetwork,
    setCpuBackend,
    setCudaBackend,
    setStdpFunction,
    backendDescription,
    resetConfiguration,
    step,
    applyStdp,
    getTargets,
    getDelays,
    getWeights,
    getPlastic,
    elapsedWallclock,
    elapsedSimulation,
    resetTimer,
    createSimulation,
    destroySimulation
};

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
