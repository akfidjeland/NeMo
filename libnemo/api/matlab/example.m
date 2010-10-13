% Simple example showing usage of Matlab bindings of nemo.
% Create a network with 1k neurons randomly connected and run it for a bit.

net = nemoNetwork;

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
	net.addNeuron(n, a, b, c, d, u, v, sigma);

	% TODO: check that direction does not matter
	sources = ones(1, 1000) * n;
	targets = 0:999;
	delays = ones(1, 1000);
	weights = 0.5 * rand(1, 1000);
	plastic = true(1, 1000);

	% Adding groups of synapses
	net.addSynapse(sources, targets, delays, weights, plastic);
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
	net.addNeuron(n, a, b, c, d, u, v, sigma);

	sources = ones(1, 1000) * n;
	targets = 0:999;
	delays = ones(1, 1000);
	weights = -rand(1, 1000);
	plastic = false(1, 1000);
	net.addSynapse(sources, targets, delays, weights, plastic);
end;


conf = nemoConfiguration;

% Set up STDP
prefire = 0.1 * exp(-(0:20)./20);
postfire = -0.08 * exp(-(0:20)./20);

conf.setStdpFunction(prefire, postfire, -1.0, 1.0);


% Run for 5s with STDP enabled
sim = nemoSimulation(net, conf);
for s=0:4
	for ms=1:1000
		fired = sim.step;
		t = s*1000 + ms;
		disp([ones(size(fired')) * t, fired'])
	end
	sim.applyStdp(1.0);
end
elapsed = sim.elapsedWallclock



% Test the synapse queries work.
%
% Note: the synapse ids returned by addSynapse should be used here. The synapse
% queries below relies on knowing the internals of how synapse ids are
% allocated (it refers to the first 10 synapse of neuron 0).

weights = sim.getWeights(0:10)
targets = sim.getTargets(0:10)
delays = sim.getDelays(0:10)
plastic = sim.getPlastic(0:10)
