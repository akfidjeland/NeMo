% nemoSetNeuron - modify one or more neurons
%  
% Synopsis:
%   nemoSetNeuron(idx, param0, param1..., state0, state1...)
%  
% Inputs:
%   idx     - neuron index
%   paramX  - neuron parameters
%   stateX  - neuron state variables
%
% The number of parameters and state variables must match the neuron type that
% was specified when the neuron was created.
%  
% The input arguments can be a mix of scalars and vectors as long as all
% vectors have the same length. Scalar arguments are replicated the appropriate
% number of times.
function setNeuron(idx, varargin)

nemo_mex(uint32(FNID), uint32(idx), varargin{:});
