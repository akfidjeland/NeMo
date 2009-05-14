% nsRun: Perform one or more simulation steps 
%
% 	F     = nsRun(NSTEPS, FSTIM)
% 	[F,T] = nsRun(NSTEPS, FSTIM)
%
% In the first form, returns an N-by-2 matrix containing the firing information
% for the next NSTEPS simulation steps on the simulation host which was
% previously initialised with nsStart.
%
% In the second form, nsRun returns the number of milliseconds that elpased
% during the host simulation in addition to the firing information.
%
% Each row in the return matrix refers to one firing event, specifying the
% time and the neuron index. The time is in the range [0, NSTEPS). In other
% words the client code has to keep track of the total time elapsed, if this
% is of interest.
%
% The firing stimulus has the same format as the return matrix, and is used to
% specify neurons which should be forced to fire at some specific time. The
% firing stimulus should be sorted by cycle number. Out-of-bounds cycle values
% or neuron indices may lead to undefined behaviour.
%
% The simulation is discrete-time, and any floating point values in FSTIM
% will be converted to uint32. 

% This is just a simple wrapper for MEX code
function [fired1, elapsed] = nsRun(nsteps, fstim1)

	global NS_SIMULATION_SOCKET;
	if ~isa(NS_SIMULATION_SOCKET, 'int32') || NS_SIMULATION_SOCKET == -1
		error 'No active simulation found'
	end;

    global NS_STDP_APPLY;
    global NS_STDP_REWARD;
    if ~isa(NS_STDP_APPLY, 'int32')
        NS_STDP_APPLY = int32(0);
        NS_STDP_REWARD = 0;
    end;

	% indexing is different (0-based vs 1-based) for both time and indices
	fstim0 = fstim1 - 1; 
	[fired0, elapsed] = nsRun_aux(NS_SIMULATION_SOCKET, nsteps, uint32(fstim0), ...
        NS_STDP_APPLY, NS_STDP_REWARD);
	fired1 = fired0 + 1;

    NS_STDP_APPLY = int32(0);
    NS_STDP_REWARD = 0;
end
