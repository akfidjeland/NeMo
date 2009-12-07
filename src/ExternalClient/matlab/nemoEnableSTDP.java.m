function nemoEnableSTDP(prefire, postfire, maxWeight, minWeight)
	global NEMO_INSTANCE;
	if exist('NEMO_INSTANCE') == 1
		if minWeight > 0
			error('minWeight is positive');
		end;
		NEMO_INSTANCE.enableStdp(prefire, postfire, maxWeight, minWeight)
	else
		error('No running instance of nemo found');
end
