% nemoStart: initialize simulation on host
%
% 	nemoStart(A, B, C, D, U, V, TARGETS, DELAYS, WEIGHTS, DT, F)
%
% Initialize a simulation (on the host specified using nemoSetHost), in order to
% allow subsequent calls to nemoRun.
%
% The network is specified using the remaining parameters. The neuron
% population are defined by A-D, U, and V which are all N-by-1 matrices, where
% N is the number of neurons in the network.
%
% The connectivity is specified using the three N-by-M matrices TARGETS, WEIGHTs
% and DELAYS. N is again the number of neurons in the network and M is the
% maximum number of synapses per neuron. If a neuron has less than M outgoing
% synapses, invalid synapses should point to neuron 0.
%
% The simulation is discrete-time, so delays are rounded to integer values.
%
% The DT specifies the number of iterations of the neuron update to run for
% each regular 1ms time step.
%
% F is the scaling factor, which is multiplied into the weight matrix.
%
% If the server is not running the function returns an error.

% This a wrapper for MEX code, with some marshalling
function nemoStart(a, b, c, d, u, v, postIdx, delays, weights, dt, F)

	% If there is already an open simulation, close it first
	global NEMO_SIMULATION_SOCKET;
	if isa(NEMO_SIMULATION_SOCKET, 'int32') && NEMO_SIMULATION_SOCKET ~= -1
		warning 'Closed already active simulation'
		nemoTerminate_aux(NEMO_SIMULATION_SOCKET);
	end;

    global NEMO_SIMULATION_HOST;
    if ~isa(NEMO_SIMULATION_HOST, 'char')
        error 'Simulation host not defined. Call nemoSetHost before nemoStart';
    end;

	global NEMO_SIMULATION_PORT;
	if ~isa(NEMO_SIMULATION_PORT, 'int32')
		NEMO_SIMULATION_PORT = 0;  % use default port on the backend
	end;

    global NEMO_STDP_ACTIVE;
    global NEMO_STDP_PRE_FIRE;
    global NEMO_STDP_POST_FIRE;
    global NEMO_STDP_MAX_WEIGHT;
    if ~isa(NEMO_STDP_ACTIVE, 'int32')
        NEMO_STDP_ACTIVE = int32(0);
        NEMO_STDP_PRE_FIRE = [];
        NEMO_STDP_POST_FIRE = [];
        NEMO_STDP_MAX_WEIGHT = 0;
    end;

    weights = F*weights;

	% All postsynaptic indices should be in the network
	checkPosts(postIdx, size(a,1));

	sd = transpose(int32(delays));
	checkDelays(sd);

	% adjust indices by one as host expects 0-based array indexing
	sp = transpose(int32(postIdx-1));
	sw = transpose(weights);
	global NEMO_SIMULATION_SOCKET;

	NEMO_SIMULATION_SOCKET = int32(nemoStart_aux(...
		NEMO_SIMULATION_HOST, NEMO_SIMULATION_PORT, ...
		a, b, c, d, u, v, sp, sd, sw, dt, ...
        NEMO_STDP_ACTIVE, ...
        NEMO_STDP_PRE_FIRE, ...
        NEMO_STDP_POST_FIRE, ...
        NEMO_STDP_MAX_WEIGHT));
end



% Check whether postsynaptic indices are out-of-bounds 
function checkPosts(posts, maxIdx)

	if ~all(all(posts >= 0))
		oob = posts(find(posts < 0));
		oob(1:min(10,size(oob,1)))
		error('Postsynaptic index matrix contains out-of-bounds members (too low). The first 10 are shown above')
	end

	if ~all(all(posts <= maxIdx)) 
		oob = posts(find(posts > maxIdx));
		oob(1:min(10,size(oob,1)))
		error('Postsynaptic index matrix contains out-of-bounds members (too high). The first 10 are shown above')
	end
end



% Check whether all delays are positive and within max
function checkDelays(delays)
	if ~all(all(delays >= 1))
		oob = delays(find(delays < 1));
		oob(1:min(10,size(oob,1)))
		error('Delay matrix contains out-of-bounds members (<1). The first 10 are shown above')
	end
	if ~all(all(delays < 32))
		oob = delays(find(delays >= 32));
		oob(1:min(10,size(oob,1)))
		error('Delay matrix contains out-of-bounds members (>=32). The first 10 are shown above')
	end
end
