% nemoGetConnectivity: get weight matrix from host
%
%	[TARGETS, DELAYS, WEIGHTS] = nemoGetConnectivity()
%
% The weights may change during simulation when using STDP. This function reads
% back the entire weight matrix, thus letting the user directly see the effect
% that STDP has (beyond a change of firing patterns).
%
% The returned matrix is the same format as that used in nemoStart, i.e. an
% N-by-M matrix, where N is the number of neurons and M is the maximum number
% of synapses per neuron.
%
% Even if the synapses are static, the weights returned by this function may
% differ slightly from the input weights given to nemoStart. This is due to
% different floating point formats on the backend.
%
% The order of synapses in the returned matrix will almost certainly differ
% from the order in the input to nemoStart. The connectivity matrix may be
% transformed in various ways as it's being mapped onto the backend. These
% transformations are not reversed. Every call to nemoGetWeights should return
% the synapses in the same order, though.

function [targets, delays, weights] = nemoGetConnectivity()

	global NEMO_SIMULATION_SOCKET;
	if ~isa(NEMO_SIMULATION_SOCKET, 'int32') || NEMO_SIMULATION_SOCKET == -1
		error 'No active simulation found'
	end;

    [t, d, w] = nemoGetConnectivity_aux(NEMO_SIMULATION_SOCKET);

    targets = transpose(t) + 1;
    delays = transpose(d);
    weights = transpose(w);
