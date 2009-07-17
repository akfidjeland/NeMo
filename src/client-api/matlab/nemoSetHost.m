% nemoSetHost: set the simulation host 
%
% 	nemoSetHost(HOSTNAME)
%
% Set the hostname to use for all subsequent calls to nemoStart in the current
% MATLAB session. 

function nemoSetHost(hostname)
    global NEMO_SIMULATION_HOST;
    NEMO_SIMULATION_HOST = hostname;
end
