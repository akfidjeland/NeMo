% nsSetPort: set the port to use to communicate with the host
%
% 	nsSetPort(PORTNUMBER)
%
% This function only needs to be called if not using the default port (56100).
% Once set, the port number is used for all subsequent host connections. 

function nsSetPort(port)
    global NS_SIMULATION_PORT;
    NS_SIMULATION_PORT = int32(port);
end
