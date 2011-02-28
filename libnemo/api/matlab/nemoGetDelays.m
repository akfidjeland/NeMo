function delays = nemoGetDelays(synapses)
% nemoGetDelays - return the conductance delays for the specified synapses
%  
% Synopsis:
%   delays = nemoGetDelays(synapses)
%  
% Inputs:
%   synapses -
%             synapse ids (as returned by addSynapse)
%    
% Outputs:
%   delays  - conductance delays of the specified synpases
%    
    delays = nemo_mex(uint32(15), uint64(synapses));
end