function weights = nemoGetWeights(synapses)
% nemoGetWeights - return the weights for the specified synapses
%  
% Synopsis:
%   weights = nemoGetWeights(synapses)
%  
% Inputs:
%   synapses -
%             synapse ids (as returned by addSynapse)
%    
% Outputs:
%   weights - weights of the specified synapses
%    
    weights = nemo_mex(uint32(16), uint64(synapses));
end