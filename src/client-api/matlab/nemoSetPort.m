% nemoSetPort: set the port to use to communicate with the host
%
% 	nemoSetPort(PORTNUMBER)
%
% This function only needs to be called if not using the default port (56100).
% Once set, the port number is used for all subsequent host connections. 

function nemoSetPort(port)
    global NEMO_SIMULATION_PORT;
    NEMO_SIMULATION_PORT = int32(port);
end
