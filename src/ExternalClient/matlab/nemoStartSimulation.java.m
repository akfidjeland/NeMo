function nemoStopSimulation
	global NEMO_INSTANCE;
	if exist('NEMO_INSTANCE') == 1
		NEMO_INSTANCE.startSimulation();
		clear NEMO_INSTANCE;
	else
		warning('No running instance of nemo found');
	end
end
