% nemoConfiguration
%  
%  
% Methods:
%     nemoConfiguration (constructor)
%     setCudaFiringBufferLength
%     cudaFiringBufferLength
%     setCudaDevice
%     setStdpFunction
%     setFractionalBits
%   
classdef nemoConfiguration < handle

	properties
		% the MEX layer keeps track of the actual pointers;
		id = -1;
	end

	methods

		function obj = nemoConfiguration()
			obj.id = nemo_mex(uint32(16));
		end

		function delete(obj)
			nemo_mex(uint32(17), obj.id);
		end

        function setCudaFiringBufferLength(obj, milliseconds)
        % setCudaFiringBufferLength - 
        %  
        % Synopsis:
        %   setCudaFiringBufferLength(milliseconds)
        %  
        % Inputs:
        %   milliseconds -
        %    
        % Set the size of the firing buffer such that it can contain a fixed
        % number of cycles worth of firing before overflowing 
            nemo_mex(uint32(18), obj.id, uint32(milliseconds));
		end

        function milliseconds = cudaFiringBufferLength(obj)
        % cudaFiringBufferLength - 
        %  
        % Synopsis:
        %   milliseconds = cudaFiringBufferLength()
        %  
        % Outputs:
        %   milliseconds -
        %             Number of milliseconds the simulation is guaranteed to be able to
        %             run before overflowing firing buffer
        %     
            milliseconds = nemo_mex(uint32(19), obj.id);
		end

        function setCudaDevice(obj, deviceNumber)
        % setCudaDevice - Set the CUDA device number to use for simulation
        %  
        % Synopsis:
        %   setCudaDevice(deviceNumber)
        %  
        % Inputs:
        %   deviceNumber -
        %    
        % The backend will choose a suitable device by default, but this
        % function can be used to override that choice 
            nemo_mex(uint32(20), obj.id, int32(deviceNumber));
		end

        function setStdpFunction(obj, prefire, postfire, minWeight, maxWeight)
        % setStdpFunction - Enable STDP and set the global STDP function
        %  
        % Synopsis:
        %   setStdpFunction(prefire, postfire, minWeight, maxWeight)
        %  
        % Inputs:
        %   prefire - STDP function values for spikes arrival times before the
        %             postsynaptic firing, starting closest to the postsynaptic firing
        %   postfire -
        %             STDP function values for spikes arrival times after the
        %             postsynaptic firing, starting closest to the postsynaptic firing
        %   minWeight -
        %             Lowest (negative) weight beyond which inhibitory synapses are not
        %             potentiated
        %   maxWeight -
        %             Highest (positive) weight beyond which excitatory synapses are not
        %             potentiated
        %    
        % The STDP function is specified by providing the values sampled at
        % integer cycles within the STDP window. 
            nemo_mex(...
                    uint32(21),...
                    obj.id,...
                    double(prefire),...
                    double(postfire),...
                    double(minWeight),...
                    double(maxWeight)...
            );
		end

        function setFractionalBits(obj, bits)
        % setFractionalBits - Set number of fractional bits used for fixed-point weight format
        %  
        % Synopsis:
        %   setFractionalBits(bits)
        %  
        % Inputs:
        %   bits    - Number of fractional bits
        %    
        % The backend uses a fixed-point number representation for weights.
        % By default the backend chooses an appropriate number of fractional
        % bits (based on the range of weights present in the network). The
        % user can call this function the force a specific number of
        % fractional bits to be used. The number of fractional bits should be
        % less than 32. 
            nemo_mex(uint32(22), obj.id, uint32(bits));
		end
	end
end
