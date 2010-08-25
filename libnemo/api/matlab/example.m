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
	targets = 0:999;
	delays = ones(1, 1000);
	weights = 0.5 * rand(1, 1000);
	plastic = true(1, 1000);

	% Adding groups of synapses
	% These can also be added individually using addSynapse
	net.addSynapses(n, targets, delays, weights, plastic);
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

	targets = 0:999;
	delays = ones(1, 1000);
	weights = -rand(1, 1000);
	plastic = false(1, 1000);
	net.addSynapses(n, targets, delays, weights, plastic);
end;


conf = nemoConfiguration;

% Set up STDP
prefire = 0.1 * exp(-(0:20)./20);
postfire = -0.08 * exp(-(0:20)./20);

conf.setStdpFunction(prefire, postfire, -1.0, 1.0);


% Run for 10s with STDP enabled
sim = nemoSimulation(net, conf);
for s=1:5
	for t=1:1000
		sim.step;
	end
	sim.applyStdp(1.0);
	[cycles, neurons] = sim.readFiring()
end
elapsed = sim.elapsedWallclock


% Read back synapses of neuron 500.
[targets, delays, weights, plastic] = sim.getSynapses(500);
