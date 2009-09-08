namespace java nemo


# The source neuron for a synapse is implicit. They are always stored with the
# presynaptic neuron
struct Synapse {
	1:i32 target,
	2:i16 delay = 1,
	3:double weight
}


typedef list<Synapse> Axon


struct IzhNeuron {
    1:i32 index,
	2:double a,
	3:double b,
	4:double c = -65.0,
	5:double d,
	6:double u,
	7:double v = -65.0,
	8:Axon axon
}



# Stimulus for a single cycle
struct Stimulus {
	1:list<i32> firing
}


# Firing data for a single cycle
typedef list<i32> Firing


exception ConstructionError {
	1:string msg
}


struct PipelineLength {
	1:i32 input,
	2:i32 output
}


service NemoFrontend {

	void setBackend(1:string host),

	void enableStdp(1:list<double> prefire,
			2:list<double> postfire,
			3:double maxWeight),

	void enablePipelining(),

	PipelineLength pipelineLength(),

	void disableStdp(),

	void addNeuron(1:IzhNeuron neuron)
		throws (1:ConstructionError err),

	void startSimulation()

	# Run simulation for multiple cycles
	list<Firing> run(1:list<Stimulus> stim)
		throws (1:ConstructionError err),

	# Run simulation for a single cycle
	#Firing step(1:Stimulus stim)
	#	throws (1:ConstructionError err),

	void applyStdp(1:double reward)
		throws (1:ConstructionError err),

	map<i32, Axon> getConnectivity(),

	void stopSimulation(),

	# Clear all state in the client. This will also stop any simulation.
	void reset(),

	# Terminate the client
	oneway void terminate()
}





service NemoBackend {

	# The backend should not have to do any building. We really want to send a
	# complete network. However, sending it in blocks of a few thousands of
	# elements is faster. The cluster in the function name does not denote any
	# hierarchical ordering.
	void addCluster(1:list<IzhNeuron> cluster),

	void addNeuron(1:IzhNeuron neuron)
		throws (1:ConstructionError err),

	# STDP is disabled by default. If STDP should be used in the simulation,
	# enableSTDP must be called before the network is finalised
	void enableStdp(
            1:list<double> prefire,
			2:list<double> postfire,
			3:double maxWeight),

	# Pipelining should be enabled before construction
	void enablePipelining(),

	# After simulation has started the user can query the actual pipeline length.
	PipelineLength pipelineLength(),

	# When the network is complete, we need to indicate network completion.
	# Calling this function is not strictly necessary, as the first instance of
	# a simulation command will do the same. However, setting up the simulation
	# may be quite time consuming, so the user is given the option of
	# controlling when this happens manually.
	void startSimulation()

	# Run simulation for multiple cycles
	list<Firing> run(1:list<Stimulus> stim)
		throws (1:ConstructionError err),

	# Run simulation for a single cycle
	#Firing step(1:Stimulus stim)
	#	throws (1:ConstructionError err),

	void applyStdp(1:double reward)
		throws (1:ConstructionError err),

	map<i32, Axon> getConnectivity(),

	oneway void stopSimulation(),
}
