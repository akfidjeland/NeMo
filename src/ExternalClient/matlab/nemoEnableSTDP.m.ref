function nemoEnableSTDP(prefire, postfire, maxWeight, minWeight)

	global NEMO_CM;
	global NEMO_STDP_ENABLED;
	global NEMO_STDP_PREFIRE;
	global NEMO_STDP_POSTFIRE;
	global NEMO_STDP_MAX_WEIGHT;
	global NEMO_STDP_MIN_WEIGHT;

	NEMO_STDP_ENABLED = true;
	NEMO_STDP_PREFIRE = prefire;
	NEMO_STDP_POSTFIRE = postfire;
	NEMO_STDP_MAX_WEIGHT = maxWeight;
	NEMO_STDP_MIN_WEIGHT = minWeight;

	if ~size(NEMO_CM)
		error('STDP enabled before network is set');
	end;
end
