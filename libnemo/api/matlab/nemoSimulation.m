% nemoSimulation
%  
% A simulation is created from a network and a configuration object.
% The simulation is run by stepping through it, providing stimulus as
% appropriate. The firing data can be read back separately (using
% readFiring) from a firing buffer which is maintained within the
% simulation itself. In the current version, some care must be taken
% to avoid overflowing this buffer.
%  
% Methods:
%     nemoSimulation (constructor)
%     step
%     applyStdp
%     getSynapses
%     elapsedWallclock
%     elapsedSimulation
%     resetTimer
%   
classdef nemoSimulation < handle

    properties
        % the MEX layer keeps track of the actual pointers;
        id = -1;
    end

    methods

        function obj = nemoSimulation(net, conf)
        % nemoSimulation - create a new simulation
        %
        % Inputs:
        %	net - an existing, populated network (nemoNetwork)
        %	conf - simulation configuration (nemoConfiguration)
        	obj.id = nemo_mex(uint32(6), uint32(net.id), uint32(conf.id));
        end

        function delete(obj)
            nemo_mex(uint32(7), obj.id);
        end
        % step - run simulation for a single cycle (1ms)
        %
        % Synpopsis:
        %	fired = step()
        %	fired = step(fstim)
        %
        % Inputs:
        % 	 fstim - An optional list of neurons, which will be forced to fire this cycle
        %
        % Output:
        %	fired - A list of the neurons which fired this cycle
        
        function fired = step(sim, fstim)
            if nargin < 2
                fired = nemo_mex(uint32(8), sim.id, uint32(zeros(1, 0)));
            else
                fired = nemo_mex(uint32(8), sim.id, uint32(fstim));
            end
        end


        function applyStdp(obj, reward)
        % applyStdp - Update synapse weights using the accumulated STDP statistics
        %  
        % Synopsis:
        %   applyStdp(reward)
        %  
        % Inputs:
        %   reward  - Multiplier for the accumulated weight change
        %     
            nemo_mex(uint32(9), obj.id, double(reward));
        end

        function [targets, delays, weights, plastic] = getSynapses(obj, source)
        % getSynapses - return synapses for a single neuron
        %  
        % Synopsis:
        %   [targets, delays, weights, plastic] = getSynapses(source)
        %  
        % Inputs:
        %   source  - index of source neuron
        %    
        % Outputs:
        %   targets - indices of target neurons
        %   delays  - conductance delay for each synapse
        %   weights -
        %   plastic - per-neuron boolean specifying whether synapse is static
        %    
        % The order of synapses returned by this function will almost
        % certainly differ from the order in which they were specified during
        % network construction. However, each call to getSynapses should
        % return the synapses in the same order. 
            [targets, delays, weights, plastic] = nemo_mex(uint32(10), obj.id, uint32(source));
        end

        function elapsed = elapsedWallclock(obj)
        % elapsedWallclock - 
        %  
        % Synopsis:
        %   elapsed = elapsedWallclock()
        %  
        % Outputs:
        %   elapsed - Return number of milliseconds of wall-clock time elapsed
        %             since first simulation step (or last timer reset)
        %     
            elapsed = nemo_mex(uint32(11), obj.id);
        end

        function elapsed = elapsedSimulation(obj)
        % elapsedSimulation - 
        %  
        % Synopsis:
        %   elapsed = elapsedSimulation()
        %  
        % Outputs:
        %   elapsed - Return number of milliseconds of simulation time elapsed
        %             since first simulation step (or last timer reset)
        %     
            elapsed = nemo_mex(uint32(12), obj.id);
        end

        function resetTimer(obj)
        % resetTimer - reset both wall-clock and simulation timer
        %  
        % Synopsis:
        %   resetTimer()
        %   
            nemo_mex(uint32(13), obj.id);
        end
    end
end
