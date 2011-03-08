function val = nemoGetNeuronParameter(idx, varno)
% nemoGetNeuronParameter - get neuron parameter
%  
% Synopsis:
%   val = nemoGetNeuronParameter(idx, varno)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%    
% Outputs:
%   val     - value of the neuron parameter
%    
% The neuron parameters do not change during simulation. For the
% Izhikevich model: 0=a, 1=b, 2=c, 3=d.
    val = nemo_mex(uint32(28), uint32(idx), uint32(varno));
end