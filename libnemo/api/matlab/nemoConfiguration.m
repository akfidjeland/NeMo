% nemoConfiguration
%  
%  
% Methods:
%     nemoConfiguration (constructor)
%     setCudaFiringBufferLength
%     cudaFiringBufferLength
%     setCudaDevice
%     setStdpFunction
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
			nemo(uint32(17), obj.id);
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
            nemo(uint32(18), obj.id, uint32(milliseconds));
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
            milliseconds = nemo(uint32(19), obj.id);
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
            nemo(uint32(20), obj.id, int32(deviceNumber));
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
        %             Highest (postivie) weight beyond which excitatory synapses are not
        %             potentiated
        %    
        % The STDP function is specified by providing the values sampled at
        % integer cycles within the STDP window. 
            nemo(...
                    uint32(21),...
                    obj.id,...
                    double(prefire),...
                    double(postfire),...
                    double(minWeight),...
                    double(maxWeight)...
            );
		end
	end
end
