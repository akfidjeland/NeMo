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
			nemo_mex(uint32(7), obj.id);
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
        % Inputs:
        %    reward - Multiplier for the accumulated weight change
        %

			nemo_mex(uint32(9), obj.id, double(reward));
		end
		function fired = readFiring(obj)
		% readFiring - return all buffered firing data
		%
		% Outputs:
		%   fired - 2 x n matrix where each row consist of a (cycle, neuron_index) pair. 
		    fired = nemo_mex(uint32(10), obj.id)
		end


		function flushFiringBuffer(obj)
        % flushFiringBuffer - 
        %
        %
        % If the user is not reading back firing, the firing output buffers
        % should be flushed to avoid buffer overflow. The overflow is not
        % harmful in that no memory accesses take place outside the buffer,
        % but an overflow may result in later calls to readFiring returning
        % non-sensical results.

			nemo_mex(uint32(11), obj.id);
		end
		% getSynapses - return synapse data for a single neuron
		%
		% Synopsis:
		%
		%	[targets, delays, weights, plastic] = getSynapses(source)
		%
		% Input:
		%   source - index of source neuron
		%
		% Outputs:
		%	targets - indices of target neurons
		%	delays - conductance delay for each synapse
		%	weights
		%	plastic - per-neuron boolean specifying whether synapse is static. 
		%
		% The order of synapses returned by this function will almost certainly differ
		% from the order in which they were specified during network construction.
		% However, each call to getSynapses should return the synapses in the same
		% order.
		
		function [targets, delays, weights, plastic] = getSynapses(obj, source)
			[targets, delays, weights, plastic] = nemo_mex(uint32(12), obj.id, uint32(source));
		end


		function elapsed = elapsedWallclock(obj)
        % elapsedWallclock - 
        %
        %

			elapsed = nemo_mex(uint32(13), obj.id);
		end

		function elapsed = elapsedSimulation(obj)
        % elapsedSimulation - 
        %
        %

			elapsed = nemo_mex(uint32(14), obj.id);
		end

		function resetTimer(obj)
        % resetTimer - reset both wall-clock and simulation timer
        %
        %

			nemo_mex(uint32(15), obj.id);
		end
	end
end
