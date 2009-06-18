% function [a, b, c, d, u, v, post, delays, weights] = smallworld()
%
% Creates a small-world network of Izhikevich neurons with a cluster-based
% topology. Simulation based on Izhikevich's program spnet.m

% function [a, b, c, d, u, v, post, delays, s] = build(Plong, F, M, Q, R, Dmax)
function [a, b, c, d, u, v, post, delays, s] = smallworld(F, seed)

rand('state', seed);

% F = 34;        % scaling factor
Plong = 0.01;  % probability of rewiring
% Plong = 0.0
Dinhb = 1;     % delay for inhibitory connections
Se = 0.7;      % initial excitatory weight
Si = -2;       % inhibitory weight
Ie = 2.4;
Ii = 2;

M = 20;                % synapses per neuron
Q = 8;                 % number of clusters
% Q = 1
R = 100;               % excitatory neurons per cluster
% R = 800
Dmax = 20;             % maximum conduction delay 
Smax = 1;              % maximum synaptic strength

% excitatory neurons      % inhibitory neurons        % total number 
Ne = Q*R;                 Ni = Ne/4;                  N = Ne+Ni;

re = rand(Ne,1);
ri = rand(Ni,1);

a = [0.02*ones(Ne,1);     0.02+0.08*ri];
b = [0.2*ones(Ne,1);      0.25-0.05*ri];
c = [-65+15*re.^2;        -65*ones(Ni,1)];
d = [8-6*re.^2;           2*ones(Ni,1)];

s = [F*Se*rand(Ne,M);     F*Si*rand(Ni,M)];   % synaptic weights
sd = zeros(N,M);                              % their derivatives

post = [zeros(Ne,M) ; zeros(Ni,M)];

% Random connectivity within clusters
for i=1:Q
   for j=1:R
      cs = (i-1)*R; % cluster start
      nn = cs+j; % neuron number
      ics = Ne+(i-1)*(Ni/Q); % inhibitory cluster start
      % Inhibitory connections
      for k=1:M/5
         post(nn,k) = ics+ceil(rand*Ni/Q); % focal
		 delays(nn,k) = Dinhb;
         % post(nn,k) = Ne+ceil(rand*Ni); % diffuse
      end
      % Excitatory connections
      for k=M/5+1:M
         post(nn,k) = cs+ceil(rand*R);
		 delays(nn,k) = ceil(rand*Dmax);
      end
   end
end

% Inhibitory connections into clusters
for i=Ne+1:N
   cc = ceil((i-Ne)/(Ni/Q));
   for j=1:M
      post(i,j) = (cc-1)*R+ceil(rand*R); % focal
      % post(i,j) = ceil(rand*Ne); % diffuse
      delays(i,j) = Dinhb;
   end
end

% Long-distance rewiring
rewired = sparse(N,N);
for i=1:Q
   for j=1:R
      cs1 = (i-1)*R; % cluster start
      nn1 = cs1+j; % neuron number
      for k=M/5+1:M
         % Rewire established excitatory connections with probability Plong
         if rand < Plong
            % Find a target for reconnection
            co = ceil(rand*(Q-1));
            cn = mod(i+co-1,Q)+1;
            cs2 = (cn-1)*R;
            nn2 = cs2+ceil(rand*R);
            post(nn1,k) = nn2;
            rewired(nn1,nn2) = 1;
         end
      end
   end
end

%{
% Excitatory delays
for i=1:Ne
   for j=1:Dmax
      delays{i,j} = [];
   end
   % Inhibitory connections
   for j=1:M/5
      delays{i,Dinhb} = [delays{i,Dinhb}, j];
   end
   % Excitatory connections
   for j=M/5+1:M
      k = ceil(rand*Dmax);
      delays{i,k} = [delays{i,k}, j];
   end
end

% Inhibitory delays
for i=Ne+1:N
   delays{i,Dinhb}=1:M;
end
%}
%{
for i=1:N
   pre{i}=find(post==i&s>0);             % pre excitatory neurons
   aux{i}=N*(Dmax-1-ceil(ceil(pre{i}/N)/(M/Dmax)))+1+mod(pre{i}-1,N);
end;
%}
% STDP = zeros(N,1001+Dmax);
v = -65*ones(N,1);                      % initial values
u = b.*v;                               % initial values

end
