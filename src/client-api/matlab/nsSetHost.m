% nsSetHost: set the simulation host 
%
% 	nsSetHost(HOSTNAME)
%
% Set the hostname to use for all subsequent calls to nsStart in the current
% MATLAB session. 

function nsSetHost(hostname)
    global NS_SIMULATION_HOST;
    NS_SIMULATION_HOST = hostname;
end
