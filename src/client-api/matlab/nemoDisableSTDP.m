% nemoDisableSTDP: disable STDP
%
%   nemoDisableSTDP
%
% Disable STDP for all subsequent simulation runs until nemoEnableSTDP is called.

function nemoDisableSTDP
    global NEMO_STDP_ACTIVE;
    global NEMO_STDP_TAU_P;
    global NEMO_STDP_TAU_D;
    global NEMO_STDP_ALPHA_P;
    global NEMO_STDP_ALPHA_D;
    global NEMO_STDP_MAX_WEIGHT;

    NEMO_STDP_ACTIVE = int32(0);
	NEMO_STDP_TAU_P = int32(0);
	NEMO_STDP_TAU_D = int32(0);
	NEMO_STDP_ALPHA_P = 0;
	NEMO_STDP_ALPHA_D = 0;
    NEMO_STDP_MAX_WEIGHT = 0;
end
