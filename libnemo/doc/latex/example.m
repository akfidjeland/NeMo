Ne = 800;
Ni = 200;
N = Ne + Ni;

re = rand(Ne,1);
nemoAddNeuron(0:Ne-1, 0.02*ones(Ne,1), 0.2*ones(Ne,1),...
	-65+15*re.^2, 8-6*re.^2, -65*0.2*ones(Ne,1),...
	-65*ones(Ne,1), 5*ones(Ne,1));
ri = rand(Ni,1);
nemoAddNeuron(Ne:Ne+Ni-1, 0.02+0.08*ri, 0.25-0.05*ri,...
	-65*ones(Ni,1), 2*ones(Ni,1), -65*0.25-0.05*ri,...
	-65*ones(Ni,1), 2*ones(Ni,1));

for n = 1:Ne-1
	nemoAddSynapse(n*ones(N,1), 0:N-1,...
		ones(N,1), 0.5*rand(N,1), false(N,1));
end

for n = Ne:N-1
	nemoAddSynapse(n*ones(N,1), 0:N-1,...
		ones(N,1), -rand(N,1), false(N,1));
end

firings = [];
nemoCreateSimulation;
for t=1:1000
	fired = nemoStep;
	firings=[firings; t+0*fired',fired'];
end
nemoDestroySimulation;
nemoClearNetwork;
plot(firings(:,1),firings(:,2),'.');
