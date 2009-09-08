% This is just a simple wrapper for MEX code
function fired1 = nemoRun(nsteps, fstim1)
	% indexing is different (0-based vs 1-based) for both time and indices
	fstim0 = fstim1 - 1; 
	fired0 = nemo_mex(mex_run, uint32(nsteps), uint32(fstim0));
	fired1 = fired0 + 1;
end
