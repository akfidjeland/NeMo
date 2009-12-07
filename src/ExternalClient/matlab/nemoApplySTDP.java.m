function nemoApplySTDP(r)
	global NEMO_INSTANCE;
	if exist('NEMO_INSTANCE') == 1
		NEMO_INSTANCE.applyStdp(r)
	else
		error('No running instance of nemo found');
	end
end
