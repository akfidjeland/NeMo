% Exercise each API function to make sure there's nothing fundementally wrong
% in the MEX layer.

nemoReset

% Configuration

try
    nemoSetCudaBackend(0);
catch e
    % A backend error here is ok. We'll get this if NeMo was compiled
    % without CUDA support, or if there are no CUDA-enabled cards on this
    % machine, OR if the CUDA runtime is not installed.
    if strcmp(e.identifier, 'nemo:backend')
        disp(e.message)
        disp('This is not neccesarily an error, though')
    else
        rethrow(e)
    end
end

% Setting the CPU backend should always work
nemoSetCpuBackend(1);

disp(nemoBackendDescription)

n = 100;
r = rand(n, 1);
a = 0.02 * ones(n, 1);
b = 0.2  * ones(n, 1);
c = -65 + 15 * r.^2;
d = 8 - 6 * r.^2;
v = -65 * ones(n, 1);
u = b .* v;
s = zeros(n, 1);

% Network construction

% Add neuron (scalar form)
nemoAddNeuron(0, a(1), b(1), c(1), d(1), u(1), v(1), s(1));

% Add neuron (vector form) 
nemoAddNeuron(1:n, a, b, c, d, u, v, s);

% The shape of the input vectors should not need to match
nemoAddNeuron(n+1:n+n, a', b, c', d, u', v, s');

% However, the length of the vectors must be the same
try
    nemoAddNeuron(2*n+1:2*n+n, a(1:10), b, c, d, u, v, s);
    error('nemo:test', 'Invalid use of addNeuron not detected correctly');
catch e
    if ~strcmp(e.identifier, 'nemo:api')
        rethrow(e);
    end   
end

testNeuronGetSet;

% TODO: set up STDP function

nemoCreateSimulation;

for ms = 1:100
    nemoStep;
end

% Simulation with firing stimulus
for ms = 1:100
    nemoStep([]);
end

% Simulation with firing stimulus
for ms = 1:100
    nemoStep([1, 2]);
end

% Simulation with current stimulus
for ms = 1:100
    nemoStep([], [1], [1.0]);
end

% Simulation with both firing and current stimulus
for ms = 1:100
    nemoStep([1], [1], [1.0]);
end

testNeuronGetSet;

nemoDestroySimulation;
