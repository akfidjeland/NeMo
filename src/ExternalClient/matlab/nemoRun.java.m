function fired1 = nemoRun(nsteps, fstim1)
	global NEMO_INSTANCE;
	if exist('NEMO_INSTANCE') == 1
		% indexing is different (0-based vs 1-based) for both time and indices
		fstim0 = reshape(fstim1 - 1, length(fstim1), 1);
		fired0 = NEMO_INSTANCE.run(nsteps, fstim0);
		fired1 = reshape(fired0 + 1, length(fired0)/2, 2);
	else
		error('No running instance of nemo found');
	end
end
