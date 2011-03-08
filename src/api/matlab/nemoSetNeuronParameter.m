function nemoSetNeuronParameter(idx, varno, val)
% nemoSetNeuronParameter - set neuron parameter
%  
% Synopsis:
%   nemoSetNeuronParameter(idx, varno, val)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%   val     - value of the neuron parameter
%    
% The neuron parameters do not change during simulation. For the
% Izhikevich model: 0=a, 1=b, 2=c, 3=d.
    nemo_mex(uint32(26), uint32(idx), uint32(varno), double(val));
end