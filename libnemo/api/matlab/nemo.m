% To use the NeMo simulator:
%
% 1. Set any global configuration options, if desired
% 2. Create a network piecemeal by adding neurons and synapses
% 3. Create simulation and step through it.
%
% For example:
%	nemoAddNeuron(...);
%	nemoAddSynapse(...);
%	nemoCreateSimulation();
%	for t in 0:999
%		fired = nemoStep();
%	end;
%	nemoDestroySimulation();
%
% The directory containing the Matlab API for NeMo also contains an example
% (example.m) showing how NeMo can be used from Matlab.
%
% The library is modal: it is either in the construction/configuration stage or
% in the simulation stage. nemoCreateSimulation switches from
% construction/configuration to simulation and nemoDestroySimulation switches
% back again. Functions are classified as either configuration, construction,
% or simulation functions, and can only be used in the appropriate stage.
%
% NeMo provides the functions listed below. See the documentation for the
% respective functions for more detail.
%
% Construction:
%  nemoAddNeuron
%  nemoAddSynapse
%  nemoNeuronCount
%  nemoClearNetwork
%
% Configuration:
%  nemoSetCpuBackend
%  nemoSetCudaBackend
%  nemoSetStdpFunction
%  nemoBackendDescription
%  nemoSetWriteOnlySynapses
%  nemoResetConfiguration
%
% Simulation:
%  nemoStep
%  nemoApplyStdp
%  nemoSetNeuron
%  nemoGetSynapsesFrom
%  nemoGetTargets
%  nemoGetDelays
%  nemoGetWeights
%  nemoGetPlastic
%  nemoElapsedWallclock
%  nemoElapsedSimulation
%  nemoResetTimer
%  nemoCreateSimulation
%  nemoDestroySimulation
%
% Others:
%  nemoReset
