% Simple example showing usage of Matlab bindings of nemo.
% Create a network with 1k neurons randomly connected and run it for a bit.

% Populate the network

% Excitatory neurons
for n=0:799
	r = rand;
	a = 0.02;
	b = 0.2;
	c = -65 + 15*r.^2;
	d = 8 - 6*r.^2;
	v = -65;
	u = b*v;
	sigma = 5 * randn;
	nemoAddNeuron(n, a, b, c, d, u, v, sigma);

	sources = ones(1, 1000) * n;
	targets = 0:999;
	delays = ones(1, 1000);
	weights = 0.5 * rand(1, 1000);
	plastic = true(1, 1000);

	% Adding groups of synapses
	nemoAddSynapse(sources, targets, delays, weights, plastic);
end;


% Inhibitory neurons
for n=800:999
	r = rand;
	a = 0.02 + 0.08 * r;
	b = 0.25 - 0.05 * r;
	c = -65;
	d = 2;
	v = -65;
	u = b * v;
	sigma = 2 * randn;
	nemoAddNeuron(n, a, b, c, d, u, v, sigma);

	sources = ones(1, 1000) * n;
	targets = 0:999;
	delays = ones(1, 1000);
	weights = -rand(1, 1000);
	plastic = false(1, 1000);
	nemoAddSynapse(sources, targets, delays, weights, plastic);
end;


% Set up STDP
prefire = 0.1 * exp(-(0:20)./20);
postfire = -0.08 * exp(-(0:20)./20);

nemoSetStdpFunction(prefire, postfire, -1.0, 1.0);


nemoCreateSimulation;

% Run for 5s with STDP enabled
for s=0:4
	for ms=1:1000
		fired = nemoStep;
		t = s*1000 + ms;
		disp([ones(size(fired')) * t, fired'])
	end
	nemoApplyStdp(1.0);
end
elapsed = nemoElapsedWallclock

% Test that the synapse queries work.
%
% Note: the synapse ids returned by addSynapse should be used here. The synapse
% queries below relies on knowing the internals of how synapse ids are
% allocated (it refers to the first 10 synapse of neuron 0).

weights = nemoGetWeights(0:9)
targets = nemoGetTargets(0:9)
delays = nemoGetDelays(0:9)
plastic = nemoGetPlastic(0:9)

nemoDestroySimulation;
