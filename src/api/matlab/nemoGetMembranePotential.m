function v = nemoGetMembranePotential(idx)
% nemoGetMembranePotential - get membane potential of a neuron
%  
% Synopsis:
%   v = nemoGetMembranePotential(idx)
%  
% Inputs:
%   idx     - neuron index
%    
% Outputs:
%   v       - membrane potential
%    
%  
% The inputs can be either all scalars or all vectors of the same
% length. The output has the same dimension as the inputs.
    v = nemo_mex(uint32(12), uint32(idx));
end