% nsTerminate: stop simulation on the server
%
% 	nsTerminate
%
% Frees up the host to cater for other clients. In the current implementation
% multiple clients can only be handled sequentially, even if there's unused
% computational capacity.

function nsTerminate()
	global NS_SIMULATION_SOCKET;
	if ~isa(NS_SIMULATION_SOCKET, 'int32')
		error 'No active simulation to terminate, or corrupt socket descriptor'
	end

	if NS_SIMULATION_SOCKET == -1
		error 'Simulation already terminated'
	end

	s = NS_SIMULATION_SOCKET;  % Change order to ensue socket cleared even if
	                           % there are errors in nsTerminate_aux.
	NS_SIMULATION_SOCKET = -1; % Valid socket handles are >= 0

	nsTerminate_aux(s);
end
