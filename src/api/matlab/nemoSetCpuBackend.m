function nemoSetCpuBackend(tcount)
% nemoSetCpuBackend - specify that the CPU backend should be used
%  
% Synopsis:
%   nemoSetCpuBackend(tcount)
%  
% Inputs:
%   tcount  - number of threads
%    
% Specify that the CPU backend should be used and optionally specify
% the number of threads to use. If the default thread count of -1 is
% used, the backend will choose a sensible value based on the
% available hardware concurrency.
    nemo_mex(uint32(4), int32(tcount));
end