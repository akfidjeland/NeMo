% This is just a simple wrapper for MEX code
function [input, output] = nemoPipelineLength()
	[input, output] = nemo_mex(mex_pipelineLength);
end
