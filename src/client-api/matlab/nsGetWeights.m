% nsGetWeights: get weight matrix from host
%
%	[TARGETS, DELAYS, WEIGHTS] = nsGetWeights()
%
% The weights may change during simulation when using STDP. This function reads
% back the entire weight matrix, thus letting the user directly see the effect
% that STDP has (beyond a change of firing patterns).
%
% The returned matrix is the same format as that used in nsStart, i.e. an
% N-by-M matrix, where N is the number of neurons and M is the maximum number
% of synapses per neuron.
%
% Even if the synapses are static, the weights returned by this function may
% differ slightly from the input weights given to nsStart. This is due to
% different floating point formats on the backend.
%
% The order of synapses in the returned matrix will almost certainly differ
% from the order in the input to nsStart. The connectivity matrix may be
% transformed in various ways as it's being mapped onto the backend. These
% transformations are not reversed. Every call to nsGetWeights should return
% the synapses in the same order, though.

function [targets, delays, weights] = nsGetWeights()

	global NS_SIMULATION_SOCKET;
	if ~isa(NS_SIMULATION_SOCKET, 'int32') || NS_SIMULATION_SOCKET == -1
		error 'No active simulation found'
	end;

    [t, d, w] = nsGetWeights_aux(NS_SIMULATION_SOCKET);

    targets = transpose(t) + 1;
    size(targets)
    delays = transpose(d);
	size(delays)
    weights = transpose(w);
	size(weights)

