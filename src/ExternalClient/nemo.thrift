
struct Synapse {
	1:i32 source,
	2:i32 target,
	3:i16 delay = 1,
	4:double weight
}


typedef list<Synapse> Axon


struct IzhNeuron {
	1:double a,
	2:double b,
	3:double c = -65.0,
	4:double d,
	5:double u,
	6:double v = -65.0,
	7:Axon axon
}


typedef map<i32, IzhNeuron> IzhNetwork


# Stimulus for a single cycle
struct Stimulus {
	1:list<i32> firing
}


# Firing data for a single cycle
typedef list<i32> Firing


exception ConstructionError {
	1:string msg
}


service NemoFrontend {

	void setBackend(1:string host),

	void setNetwork(1:IzhNetwork net)
		throws (1:ConstructionError err),

    # Run simulation for multiple cycles 
	list<Firing> run(1:list<Stimulus> stim)
		throws (1:ConstructionError err),

	void enableStdp(1:list<double> prefire,
			2:list<double> postfire,
			3:double maxWeight),

	void disableStdp(),

	void applyStdp(1:double reward)
		throws (1:ConstructionError err),

	map<i32, Axon> getConnectivity()
}
