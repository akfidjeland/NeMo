function nemoStopSimulation
	global NEMO_INSTANCE;
	if exist('NEMO_INSTANCE') == 1
		NEMO_INSTANCE.stopSimulation();
	else
		warning('No running instance of nemo found');
	end
end
