function synapses = nemoGetSynapsesFrom(source)
% nemoGetSynapsesFrom - return the synapse ids for all synapses with the given source neuron
%  
% Synopsis:
%   synapses = nemoGetSynapsesFrom(source)
%  
% Inputs:
%   source  - source neuron index
%    
% Outputs:
%   synapses -
%             synapse ids
%    
    synapses = nemo_mex(uint32(25), uint32(source));
end