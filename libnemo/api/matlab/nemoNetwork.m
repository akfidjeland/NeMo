% nemoNetwork
%  
% Networks are constructed by adding individual neurons, and single
% or groups of synapses to the network. Neurons are given indices
% (from 0) which should be unique for each neuron. When adding
% synapses the source or target neurons need not necessarily exist
% yet, but should be defined before the network is finalised.
%  
% Methods:
%     nemoNetwork (constructor)
%     addNeuron
%     addSynapse
%     addSynapses
%     neuronCount
%   
classdef nemoNetwork < handle

    properties
        % the MEX layer keeps track of the actual pointers;
        id = -1;
    end

    methods

        function obj = nemoNetwork()
        	obj.id = nemo_mex(uint32(0));
        end

        function delete(obj)
            nemo_mex(uint32(1), obj.id);
        end

        function addNeuron(obj, idx, a, b, c, d, u, v, sigma)
        % addNeuron - add a single neuron to network
        %  
        % Synopsis:
        %   addNeuron(idx, a, b, c, d, u, v, sigma)
        %  
        % Inputs:
        %   idx     - Neuron index (0-based)
        %   a       - Time scale of the recovery variable
        %   b       - Sensitivity to sub-threshold fluctuations in the membrane
        %             potential v
        %   c       - After-spike value of the membrane potential v
        %   d       - After-spike reset of the recovery variable u
        %   u       - Initial value for the membrane recovery variable
        %   v       - Initial value for the membrane potential
        %   sigma   - Parameter for a random gaussian per-neuron process which
        %             generates random input current drawn from an N(0, sigma)
        %             distribution. If set to zero no random input current will be
        %             generated
        %    
        % The neuron uses the Izhikevich neuron model. See E. M. Izhikevich
        % "Simple model of spiking neurons", IEEE Trans. Neural Networks, vol
        % 14, pp 1569-1572, 2003 for a full description of the model and the
        % parameters. 
            nemo_mex(...
                    uint32(2),...
                    obj.id,...
                    uint32(idx),...
                    double(a),...
                    double(b),...
                    double(c),...
                    double(d),...
                    double(u),...
                    double(v),...
                    double(sigma)...
            );
        end

        function addSynapse(obj, source, target, delay, weight, plastic)
        % addSynapse - add a single synapse to given neuron
        %  
        % Synopsis:
        %   addSynapse(source, target, delay, weight, plastic)
        %  
        % Inputs:
        %   source  - Index of source neuron
        %   target  - Index of target neuron
        %   delay   - Synapse conductance delay in milliseconds
        %   weight  - Synapse weights
        %   plastic - Boolean specifying whether or not this synapse is plastic
        %     
            nemo_mex(...
                    uint32(3),...
                    obj.id,...
                    uint32(source),...
                    uint32(target),...
                    uint32(delay),...
                    double(weight),...
                    uint8(plastic)...
            );
        end

        function addSynapses(obj, source, targets, delays, weights, plastic)
        % addSynapses - add multiple synapses with the same source and delay
        %  
        % Synopsis:
        %   addSynapses(source, targets, delays, weights, plastic)
        %  
        % Inputs:
        %   source  - Source neuron index
        %   targets - Vector of target indices
        %   delays  - Vector of delays (in milliseconds)
        %   weights - Vector of weights
        %   plastic - Vector of booleans specifying whether each neuron is
        %             plastic
        %    
        % The input vectors should all have the same length 
            nemo_mex(...
                    uint32(4),...
                    obj.id,...
                    uint32(source),...
                    uint32(targets),...
                    uint32(delays),...
                    double(weights),...
                    uint8(plastic)...
            );
        end

        function ncount = neuronCount(obj)
        % neuronCount - 
        %  
        % Synopsis:
        %   ncount = neuronCount()
        %  
        % Outputs:
        %   ncount  - number of neurons in the network
        %     
            ncount = nemo_mex(uint32(5), obj.id);
        end
    end
end
