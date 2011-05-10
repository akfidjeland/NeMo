% nemoAddNeuron - add a single neuron to the network
%  
% Synopsis:
%   nemoAddNeuron(type, idx, param0, param1..., state0, state1...)
%  
% Inputs:
%   type    - neuron type, as returned by nemoAddNeuronType
%   idx     - neuron index (0-based)
%   paramX  - neuron parameters
%   stateX  - neuron state variables
%
% The number of parameters and state variables must match the neuron type.
%  
% The input arguments can be a mix of scalars and vectors as long as all
% vectors have the same length. Scalar arguments are replicated the appropriate
% number of times.
function addNeuron(type, idx, varargin)

nemo_mex(uint32(FNID), uint32(type), uint32(idx), varargin{:});
