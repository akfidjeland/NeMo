#include <vector>
#include <algorithm>
#include <string>
#ifdef _WIN32
#include <win_stdint.h> // in own local build directory
#else
#include <stdint.h>
#endif

#include <boost/numeric/conversion/cast.hpp>
#include <mex.h>


#include <nemo.hpp>


/* We maintain pointers to all simulator objects as globals here, since Matlab
 * does not seem to have any support for foreign pointers.
 *
 * Instead of returning opaque poitners to the user we return indices into
 * these global vectors */ 

/* The user can specify multiple networks */
static std::vector<nemo::Network*> g_networks;

/* Additionally, the user can specify multiple configurations. */
static std::vector<nemo::Configuration*> g_configs;

//! \todo should we limit this to one?
static std::vector<nemo::Simulation*> g_sims;



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
		mexErrMsgIdAndTxt("nemo:mex", "unexpected number of input arguments");
	}

}



/* Return numeric scalar in output argument 0 after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnScalar(mxArray* plhs[], const N& val)
{
	plhs[0] = mxCreateNumericMatrix(1, 1, classId<M>(), mxREAL);
	*(static_cast<M*>(mxGetData(plhs[0]))) = val;
}



/* Return numeric vector in output argument n after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnVector(mxArray* plhs[], int argno, const std::vector<N>& vec)
{
	mxArray* ret = mxCreateNumericMatrix(vec.size(), 1, classId<M>(), mxREAL);
	std::transform(vec.begin(), vec.end(), 
			static_cast<M*>(mxGetData(ret)),
			boost::numeric_cast<N, M>);
	plhs[argno] = ret;
}


/* Converting nemo types to strings is useful for error reporting */
template<class T> const std::string collectionName() { return "unknown"; }
template<> const std::string collectionName<nemo::Network>() { return "network"; }
template<> const std::string collectionName<nemo::Simulation>() { return "simulation"; }
template<> const std::string collectionName<nemo::Configuration>() { return "configuration"; }



template<class T>
uint32_t
getHandleId(std::vector<T*> collection, const mxArray* prhs[], unsigned argno)
{
	uint32_t id = scalar<uint32_t, uint32_t>(prhs[argno]);
	if(id >= collection.size()) {
		mexErrMsgIdAndTxt("nemo:mex", "%s handle id %u out of bounds (%u items)", 
				collectionName<T>().c_str(), id, collection.size());
	}
	return id;
}



template<class T>
T*
getHandle(std::vector<T*>& collection, const mxArray* prhs[], unsigned argno)
{
	uint32_t id = getHandleId<T>(collection, prhs, argno);
	T* ptr = collection.at(id);
	if(ptr == NULL) {
		mexErrMsgIdAndTxt("nemo:mex", "%s handle id %u is NULL", collectionName<T>().c_str(), id);
	}
	return ptr;
}



//! \todo need to pass in relevant destructor here
template<class T>
void
freeCollection(std::vector<T*>& handles)
{
	for(typename std::vector<T*>::iterator i = handles.begin(); i != handles.end(); ++i) {
		if(*i != NULL) {
			delete *i;
		}
	}

}



void
freeGlobals()
{
	//! \todo add this back
	freeCollection(g_networks);
	freeCollection(g_configs);
	freeCollection(g_sims);
}



template<class T>
void
newHandle(std::vector<T*>& collection, mxArray* plhs[])
{
	/* It's possible to leave "holes" in this vector of networks, but that's
	 * unlikely to ever be a problem */
	uint32_t id = collection.size(); 
	collection.push_back(new T());
	mexAtExit(freeGlobals);
	returnScalar<uint32_t, uint32_t>(plhs, id);
}



template<class T>
void
deleteHandle(std::vector<T*>& collection, const mxArray* prhs[], unsigned argno)
{
	uint32_t id = getHandleId<T>(collection, prhs, argno);
	T* ptr = collection.at(id);
	if(ptr != NULL) {
		delete ptr;
		collection.at(id) = NULL;
	}
}



nemo::Network*
getNetwork(const mxArray* prhs[], unsigned argno)
{
	return getHandle<nemo::Network>(g_networks, prhs, argno);
}



nemo::Configuration*
getConfiguration(const mxArray* prhs[], unsigned argno)
{
	return getHandle<nemo::Configuration>(g_configs, prhs, argno);
}



nemo::Simulation*
getSimulation(const mxArray* prhs[], unsigned argno)
{
	return getHandle<nemo::Simulation>(g_sims, prhs, argno);
}



void
newNetwork(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	newHandle(g_networks, plhs);
}



void
deleteNetwork(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	deleteHandle(g_networks, prhs, 1);
}



void
newConfiguration(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	newHandle(g_configs, plhs);
}



void
deleteConfiguration(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	deleteHandle(g_configs, prhs, 1);
}



void
newSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	uint32_t id = g_sims.size(); 

	nemo::Network* net = getNetwork(prhs, 1);
	nemo::Configuration* conf = getConfiguration(prhs, 2);
	nemo::Simulation* sim = NULL;
	try {
		sim = nemo::Simulation::create(*net, *conf);
	} catch (std::runtime_error& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}

	g_sims.push_back(sim);	
	mexAtExit(freeGlobals);

	returnScalar<uint32_t, uint32_t>(plhs, id);
}



void
deleteSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	deleteHandle(g_sims, prhs, 1);
}



void
readFiring(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	try {
		const std::vector<unsigned>* cycles;
		const std::vector<unsigned>* nidx;
		getSimulation(prhs, 1)->readFiring(&cycles, &nidx);
		size_t flen = cycles->size();
		mxArray* ret = mxCreateNumericMatrix(flen, 2, classId<uint32_t>(), mxREAL);
		uint32_t* ml_cycles = (uint32_t*) mxGetData(ret);
		uint32_t* ml_nidx = ml_cycles + flen;
		std::transform(cycles->begin(), cycles->end(),
				ml_cycles, boost::numeric_cast<unsigned, uint32_t>);
		std::transform(nidx->begin(), nidx->end(),
				ml_nidx, boost::numeric_cast<unsigned, uint32_t>);
		plhs[0] = ret;
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}



void
getSynapses(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 1);

	const std::vector<unsigned>* targets;
	const std::vector<unsigned>* delays;
	const std::vector<float>* weights;
	const std::vector<unsigned char>* plastic;

	try {
		getSimulation(prhs, 1)->getSynapses(
				scalar<unsigned, uint32_t>(prhs[2]),
				&targets, &delays, &weights, &plastic);
	} catch (std::exception& e) {     
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}

	returnVector<unsigned, uint32_t>(plhs, 0, *targets);
	returnVector<unsigned, uint32_t>(plhs, 1, *delays);
	returnVector<float, double>(plhs, 2, *weights);
	returnVector<unsigned char, uint8_t>(plhs, 3, *plastic);
}
/* AUTO-GENERATED CODE START */

void
addNeuron(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 8);
	try {
		getNetwork(prhs, 1)->addNeuron(
			scalar<unsigned,uint32_t>(prhs[2]),
			scalar<float,double>(prhs[3]),
			scalar<float,double>(prhs[4]),
			scalar<float,double>(prhs[5]),
			scalar<float,double>(prhs[6]),
			scalar<float,double>(prhs[7]),
			scalar<float,double>(prhs[8]),
			scalar<float,double>(prhs[9]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
addSynapse(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 5);
	try {
		getNetwork(prhs, 1)->addSynapse(
			scalar<unsigned,uint32_t>(prhs[2]),
			scalar<unsigned,uint32_t>(prhs[3]),
			scalar<unsigned,uint32_t>(prhs[4]),
			scalar<float,double>(prhs[5]),
			scalar<unsigned char,uint8_t>(prhs[6]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
addSynapses(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 5);
	try {
		getNetwork(prhs, 1)->addSynapses(
			scalar<unsigned,uint32_t>(prhs[2]),
			vector<unsigned, uint32_t>(prhs[3]),
			vector<unsigned, uint32_t>(prhs[4]),
			vector<float, double>(prhs[5]),
			vector<unsigned char, uint8_t>(prhs[6]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
neuronCount(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 0);
	try {
		getNetwork(prhs, 1)->neuronCount();
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
step(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 1);
	try {
		getSimulation(prhs, 1)->step(vector<unsigned, uint32_t>(prhs[2]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
applyStdp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 1);
	try {
		getSimulation(prhs, 1)->applyStdp(scalar<float,double>(prhs[2]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
flushFiringBuffer(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 0);
	try {
		getSimulation(prhs, 1)->flushFiringBuffer();
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
elapsedWallclock(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 0);
	try {
		returnScalar<unsigned long, uint64_t>(plhs, getSimulation(prhs, 1)->elapsedWallclock());
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
elapsedSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 0);
	try {
		returnScalar<unsigned long, uint64_t>(plhs, getSimulation(prhs, 1)->elapsedSimulation());
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
resetTimer(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 0);
	try {
		getSimulation(prhs, 1)->resetTimer();
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
setCudaFiringBufferLength(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 1);
	try {
		getConfiguration(prhs, 1)->setCudaFiringBufferLength(scalar<unsigned,uint32_t>(prhs[2]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
cudaFiringBufferLength(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 0);
	try {
		returnScalar<unsigned, uint32_t>(plhs, getConfiguration(prhs, 1)->cudaFiringBufferLength());
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
setCudaDevice(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 1);
	try {
		getConfiguration(prhs, 1)->setCudaDevice(scalar<int,int32_t>(prhs[2]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




void
setStdpFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	checkInputCount(nrhs, 4);
	try {
		getConfiguration(prhs, 1)->setStdpFunction(
			vector<float, double>(prhs[2]),
			vector<float, double>(prhs[3]),
			scalar<float,double>(prhs[4]),
			scalar<float,double>(prhs[5]));
	} catch (std::exception& e) {
		mexErrMsgIdAndTxt("nemo:backend", e.what());
	}
}




typedef void (*fn_ptr)(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

#define FN_COUNT 22

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
	setCudaFiringBufferLength,
	cudaFiringBufferLength,
	setCudaDevice,
	setStdpFunction};

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
