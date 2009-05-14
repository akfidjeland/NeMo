% nsDisableSTDP: disable STDP
%
%   nsDisableSTDP
%
% Disable STDP for all subsequent simulation runs until nsEnableSTDP is called.
function nsDisableSTDP
    global NS_STDP_ACTIVE;
    global NS_STDP_TAU_P;
    global NS_STDP_TAU_D;
    global NS_STDP_ALPHA_P;
    global NS_STDP_ALPHA_D;
    global NS_STDP_MAX_WEIGHT;

    NS_STDP_ACTIVE = int32(0);
	NS_STDP_TAU_P = int32(0);
	NS_STDP_TAU_D = int32(0);
	NS_STDP_ALPHA_P = 0;
	NS_STDP_ALPHA_D = 0;
    NS_STDP_MAX_WEIGHT = 0;
end
