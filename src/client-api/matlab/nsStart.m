% nsStart: initialize simulation on host
%
% 	nsStart(A, B, C, D, U, V, SPOST, SDELAY, SWEIGHT, DT, F)
%
% Initialize a simulation (on the host specified using nsSetHost), in order to
% allow subsequent calls to nsRun.
%
% The network is specified using the remaining parameters. The neuron
% population are defined by A-D, U, and V which are all N-by-1 matrices, where
% N is the number of neurons in the network.
%
% The connectivity is specified using the three N-by-M matrices SPOST, SWEIGHT
% and SDELAY. N is again the number of neurons in the network and M is the
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
function nsStart(a, b, c, d, u, v, postIdx, delays, weights, dt, F)

	% If there is already an open simulation, close it first
	global NS_SIMULATION_SOCKET;
	if isa(NS_SIMULATION_SOCKET, 'int32') && NS_SIMULATION_SOCKET ~= -1
		warning 'Closed already active simulation'
		nsTerminate_aux(NS_SIMULATION_SOCKET);
	end;

    global NS_SIMULATION_HOST;
    if ~isa(NS_SIMULATION_HOST, 'char')
        error 'Simulation host not defined. Call nsSetHost before nsStart';
    end;

	global NS_SIMULATION_PORT;
	if ~isa(NS_SIMULATION_PORT, 'int32')
		NS_SIMULATION_PORT = 0;  % use default port on the backend
	end;

    global NS_STDP_ACTIVE;
    global NS_STDP_PRE_FIRE;
    global NS_STDP_POST_FIRE;
    global NS_STDP_MAX_WEIGHT;
    if ~isa(NS_STDP_ACTIVE, 'int32')
        NS_STDP_ACTIVE = int32(0);
        NS_STDP_PRE_FIRE = [];
        NS_STDP_POST_FIRE = [];
        NS_STDP_MAX_WEIGHT = 0;
    end;

    weights = F*weights;

	% All postsynaptic indices should be in the network
	checkPosts(postIdx, size(a,1));

	sd = transpose(int32(delays));
	checkDelays(sd);

	% adjust indices by one as host expects 0-based array indexing
	sp = transpose(int32(postIdx-1));
	sw = transpose(weights);
	global NS_SIMULATION_SOCKET;

	NS_SIMULATION_SOCKET = int32(nsStart_aux(...
		NS_SIMULATION_HOST, NS_SIMULATION_PORT, ...
		a, b, c, d, u, v, sp, sd, sw, dt, ...
        NS_STDP_ACTIVE, ...
        NS_STDP_PRE_FIRE, ...
        NS_STDP_POST_FIRE, ...
        NS_STDP_MAX_WEIGHT));
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
