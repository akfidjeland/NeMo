% nemoTerminate: stop simulation on the server
%
% 	nemoTerminate
%
% Frees up the host to cater for other clients. In the current implementation
% multiple clients can only be handled sequentially, even if there's unused
% computational capacity.

function nemoTerminate()
	global NEMO_SIMULATION_SOCKET;
	if ~isa(NEMO_SIMULATION_SOCKET, 'int32')
		error 'No active simulation to terminate, or corrupt socket descriptor'
	end

	if NEMO_SIMULATION_SOCKET == -1
		error 'Simulation already terminated'
	end

	s = NEMO_SIMULATION_SOCKET;  % Change order to ensure socket cleared even if
	                           % there are errors in nemoTerminate_aux.
	NEMO_SIMULATION_SOCKET = -1; % Valid socket handles are >= 0

	nemoTerminate_aux(s);
end
