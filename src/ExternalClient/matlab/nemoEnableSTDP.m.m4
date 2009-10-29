function nemoEnableSTDP(prefire, postfire, maxWeight, minWeight)
	if minWeight > 0
		error('minWeight is positive');
	end;
	nemo_mex(mex_enableSTDP, prefire, postfire, maxWeight, minWeight);
end
