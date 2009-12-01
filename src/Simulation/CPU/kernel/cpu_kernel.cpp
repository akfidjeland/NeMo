extern "C" {
#include "cpu_kernel.h"
}

#include <string>
#include <stdexcept>

#include <STDP.hpp>

#include "Network.hpp"


/* To be able to do error handling in a thread-safe manner in the C interface,
 * without storing error strings/codes in the network itself use a simple
 * wrapper. Calls the the underlying network should be wrapped in a try/catch
 * block and the error here set accordingly. The user can then get back an
 * error string later. It might be possible to do this more elegantly using
 * some form of smart pointer, but brute-forcing it with preprocessor hacks
 * works fine as well. */
struct network_t {

	public:

		network_t() :
				status(STATUS_OK) {
			m_network = new nemo::cpu::Network();
		}

		network_t(nemo::cpu::Network* net) :
			status(STATUS_OK), m_network(net) {}

		~network_t() {
			delete m_network;
		}

		nemo::cpu::Network* m_network;

		void setError(const char* msg) {
			error_msg = std::string(msg);
			status = STATUS_ERROR;
		}

		std::string error_msg;
		status_t status;

	private:

		// don't allow copying network_t
		network_t(const network_t&);
		network_t& operator=(const network_t&);
};

// in the c interface, typedef network_t* NETWORK


#define SAFE_CALL(ptr, call) {                                                \
        try {                                                                 \
            static_cast<network_t*>(ptr)->m_network->call;                    \
        } catch (std::exception& e) {                                         \
            static_cast<network_t*>(ptr)->setError(e.what());                 \
        }                                                                     \
        return static_cast<network_t*>(ptr)->status;                          \
    }


// call function without handling exceptions
#define UNSAFE_CALL(ptr, call) static_cast<network_t*>(ptr)->m_network->call



NETWORK
cpu_new_network()
{
	return new network_t();
}



NETWORK
cpu_set_network(fp_t a[],
		fp_t b[],
		fp_t c[],
		fp_t d[],
		fp_t u[],
		fp_t v[],
		fp_t sigma[], //set to 0 if not thalamic input required
		size_t ncount)
{
	return new network_t(new nemo::cpu::Network(a, b, c, d, u, v, sigma, ncount));
}



status_t
cpu_add_neuron(NETWORK net,
		nidx_t idx,
		fp_t a, fp_t b, fp_t c, fp_t d,
		fp_t u, fp_t v, fp_t sigma)
{
	SAFE_CALL(net, addNeuron(idx, a, b, c, d, u, v, sigma));
}



status_t
cpu_add_synapses(NETWORK net,
		nidx_t source,
		delay_t delay,
		nidx_t targets[],
		weight_t weights[],
		unsigned int is_plastic[],
		size_t length)
{
	SAFE_CALL(net, addSynapses(source, delay, targets, weights, is_plastic, length));
}


status_t
cpu_enable_stdp(NETWORK network,
		size_t pre_len,
		size_t post_len,
		double* pre_fn,
		double* post_fn,
		double w_max,
		double w_min)
{
	nemo::STDP<double> conf;
	nemo::configure_stdp<double>(conf, pre_len, post_len, pre_fn, post_fn, w_max, w_min);
	SAFE_CALL(network, configureStdp(conf));
}


status_t
cpu_start_simulation(NETWORK network)
{
	SAFE_CALL(network, startSimulation());
}



status_t
cpu_step(NETWORK network, unsigned int fstim[])
{
	SAFE_CALL(network, step(fstim));
}



status_t
cpu_deliver_spikes(NETWORK network)
{
	SAFE_CALL(network, deliverSpikes());
}



status_t
cpu_update(NETWORK network, unsigned int fstim[])
{
	SAFE_CALL(network, update(fstim));
}



status_t
cpu_read_firing(NETWORK network,
		unsigned int** neuronIdx,
		unsigned int* nfired)
{
	try {
		const std::vector<unsigned int>& firings =
			UNSAFE_CALL(network, readFiring());
		*neuronIdx = const_cast<unsigned int*>(&firings[0]);
		*nfired = firings.size();
		return STATUS_OK;
	} catch (std::exception& e) {
		static_cast<network_t*>(network)->setError(e.what());
		*neuronIdx = NULL;
		*nfired = 0;
		return STATUS_ERROR;
	}
}



status_t
cpu_apply_stdp(NETWORK network, double reward)
{
	SAFE_CALL(network, applyStdp(reward));
}



long int
cpu_elapsed_ms(NETWORK network)
{
	return UNSAFE_CALL(network, elapsed());
}



status_t
cpu_reset_timer(NETWORK network)
{
	SAFE_CALL(network, resetTimer());
}



void
cpu_delete_network(NETWORK network)
{
	delete static_cast<network_t*>(network);
}


const char*
cpu_last_error(NETWORK network)
{
	return static_cast<network_t*>(network)->error_msg.c_str();
}



