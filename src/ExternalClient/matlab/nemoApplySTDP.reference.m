function nemoApplySTDP(reward)

	global NEMO_CM;
	global NEMO_STDP_ACC;
	global NEMO_STDP_MAX_WEIGHT;
	global NEMO_STDP_MIN_WEIGHT;
	global NEMO_RTS_STDP;
	global NEMO_RCM;
	%global NEMO_CYCLE;
	global NEMO_RTS_ENABLED;
	global NEMO_RCM_CHANGED;

	verbose = false;
	if(verbose)
		fid = fopen('stdp.dat','wt');
	end

	[posts, ds] = find(NEMO_RCM_CHANGED);
	% TODO: may be able to do this better
	NEMO_RCM_CHANGED = 0;

	for i=1:length(posts)

		post = posts(i);
		d = ds(i);

		w_diff = NEMO_STDP_ACC{post,d};

		changed = (w_diff ~= 0);
		NEMO_STDP_ACC{post,d}(changed) = 0;

		if ~any(changed) || reward == 0
			continue;
		end

		pres = NEMO_RCM{post,d}(changed);

		w_init = NEMO_CM{d}(post, pres);

		% TODO: store ACC in same format as input
		if reward == 1
			w_new = w_init + w_diff(changed);
		else % we already know it's not 0
			w_new = w_init + reward * w_diff(changed);
		end

		w_new(sign(w_new) ~= sign(w_init)) = 0;
		w_new(w_new > NEMO_STDP_MAX_WEIGHT) = NEMO_STDP_MAX_WEIGHT;
		w_new(w_new < NEMO_STDP_MIN_WEIGHT) = NEMO_STDP_MIN_WEIGHT;

		NEMO_CM{d}(post,pres) = w_new;

		% TODO: add back logging
		%{
		if(verbose)
			fprintf(fid, 'c%u: stdp (%u->%u) %f %+f = %f\n',...
				NEMO_CYCLE, pre, post, w_init, w_diff(changed), w_new);
		end
		%}

		if NEMO_RTS_ENABLED
			NEMO_RTS_STDP = NEMO_RTS_STDP + length(find(changed));
		end
	end
end
