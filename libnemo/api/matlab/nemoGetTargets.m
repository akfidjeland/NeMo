function targets = nemoGetTargets(synapses)
% nemoGetTargets - return the targets for the specified synapses
%  
% Synopsis:
%   targets = nemoGetTargets(synapses)
%  
% Inputs:
%   synapses -
%             synapse ids (as returned by addSynapse)
%    
% Outputs:
%   targets - indices of target neurons
%    
    targets = nemo_mex(uint32(11), uint64(synapses));
end