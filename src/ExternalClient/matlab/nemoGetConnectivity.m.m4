function [targets, delays, weights] = nemoGetConnectivity()
    [targets, delays, weights] = nemo_mex(mex_getConnectivity);
end
