function id = nemoAddSynapse(source, target, delay, weight, plastic)
% nemoAddSynapse - add a single synapse to the network
%  
% Synopsis:
%   id = nemoAddSynapse(source, target, delay, weight, plastic)
%  
% Inputs:
%   source  - Index of source neuron
%   target  - Index of target neuron
%   delay   - Synapse conductance delay in milliseconds
%   weight  - Synapse weights
%   plastic - Boolean specifying whether or not this synapse is plastic
%    
% Outputs:
%   id      - Unique synapse ID
%    
%  
% The inputs can be either all scalars or all vectors of the same
% length. The output has the same dimension as the inputs.
    id = nemo_mex(...
                 uint32(1),...
                 uint32(source),...
                 uint32(target),...
                 uint32(delay),...
                 double(weight),...
                 uint8(plastic)...
         );
end