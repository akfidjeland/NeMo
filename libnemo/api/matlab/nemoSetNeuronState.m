function nemoSetNeuronState(idx, varno, val)
% nemoSetNeuronState - set neuron state variable
%  
% Synopsis:
%   nemoSetNeuronState(idx, varno, val)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%   val     - value of the relevant variable
%    
% For the Izhikevich model: 0=u, 1=v
    nemo_mex(uint32(25), uint32(idx), uint32(varno), double(val));
end