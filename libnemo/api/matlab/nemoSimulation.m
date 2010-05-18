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
%     readFiring
%     flushFiringBuffer
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
		%	net - an existing, populated network (nemoNetwork)
		%	conf - simulation configuration (nemoConfiguration)
			obj.id = nemo_mex(uint32(6), uint32(net.id), uint32(conf.id));
		end

		function delete(obj)
			nemo(uint32(7), obj.id);
		end
		% step - run simulation for a single cycle (1ms)
		%
		% Synpopsis:
		%	step()
		%	step(fstim)
		%
		% Inputs:
		% 	 fstim - An optional list of neurons, which will be forced to fire this cycle
		
		function step(sim, fstim)
		    if nargin < 2
		    	nemo_mex(uint32(8), sim.id, uint32(zeros(1, 0)));
		    else
		    	nemo_mex(uint32(8), sim.id, uint32(fstim));
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
            nemo(uint32(9), obj.id, double(reward));
		end

        function [cycles, nidx] = readFiring(obj)
        % readFiring - read all buffered firing data
        %  
        % Synopsis:
        %   [cycles, nidx] = readFiring()
        %  
        % Outputs:
        %   cycles  - Cycles during which firings took place
        %   nidx    - Neurons which fired
        %    
        % Firing data is buffered in the simulation while the simulation is
        % running. readFiring reads all the data that has been buffered since
        % the previous call to this function (or the start of the simulation
        % of this is the first call. The return vectors are valid until the
        % next call to this function. 
            [cycles, nidx] = nemo(uint32(10), obj.id);
		end

        function flushFiringBuffer(obj)
        % flushFiringBuffer - 
        %  
        % Synopsis:
        %   flushFiringBuffer()
        %  
        % If the user is not reading back firing, the firing output buffers
        % should be flushed to avoid buffer overflow. The overflow is not
        % harmful in that no memory accesses take place outside the buffer,
        % but an overflow may result in later calls to readFiring returning
        % non-sensical results. 
            nemo(uint32(11), obj.id);
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
            [targets, delays, weights, plastic] = nemo(uint32(12), obj.id, uint32(source));
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
            elapsed = nemo(uint32(13), obj.id);
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
            elapsed = nemo(uint32(14), obj.id);
		end

        function resetTimer(obj)
        % resetTimer - reset both wall-clock and simulation timer
        %  
        % Synopsis:
        %   resetTimer()
        %   
            nemo(uint32(15), obj.id);
		end
	end
end
