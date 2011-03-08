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
    v = nemo_mex(uint32(12), uint32(idx));
end