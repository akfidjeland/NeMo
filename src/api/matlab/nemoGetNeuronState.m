function val = nemoGetNeuronState(idx, varno)
% nemoGetNeuronState - get neuron state variable
%  
% Synopsis:
%   val = nemoGetNeuronState(idx, varno)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%    
% Outputs:
%   val     - value of the relevant variable
%    
% For the Izhikevich model: 0=u, 1=v.
%  
% The inputs can be either all scalars or all vectors of the same
% length. The output has the same dimension as the inputs.
    val = nemo_mex(uint32(27), uint32(idx), uint32(varno));
end