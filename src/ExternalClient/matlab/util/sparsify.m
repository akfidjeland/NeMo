% Store one sparse matrix per delay
function ret = sparsify(targets, delays, weights, d_max, warnDuplicates)

	ret = cell(d_max, 1);

	[n,s] = size(targets);                     % n: number of neurons
	                                           % s: number of synapses per neuron
	len = n*s;

	%invalid = find(targets == 0);
	%weights(targets == 0) = 0;

	% We can use the same indices for every delay-specific CM
	j = reshape(targets, len, 1);              % postsynaptic indices
	i = reshape(repmat([1:n]', s, 1), len, 1); % presynaptic indices
	wvec = reshape(weights, len, 1);           % synaptic weights
	dvec = reshape(delays, len, 1);            % conducatance delays

	% The user may have set targets to 0 for invalid synapses. However,
	% 'sparse' below needs all entries to be positive. We must therefore strip
	% these out.
	valid = find(j ~= 0); % (negative indices are errors)
	j = j(valid);
	i = i(valid);
	wvec = wvec(valid);
	dvec = dvec(valid);

	% In the input format invalid synapses are specified in the 'targets'
	% matrix, but in the sparse matrix construction below, invalid synapses are
	% specified in the data field, so we need to clear these entries.
	%wvec(j==0) = 0;


	for d = 1:d_max

		% mask out any delays not currently under consideration
		w_tmp = wvec;
		w_tmp(dvec ~= d) = 0;

		% It's possible to have synapses with the same pre, post, and delay. We
		% need to add their weights to get this reference implementation to
		% work. This is already done by 'sparse'. 

		ret{d} = sparse(j, i, w_tmp, n, n);

		% The presence of the above-mentioned duplicates will cause some
		% divergence between the reference implementation and the GPU
		% implementation. We issue a warning, to at least let the user know. 
		if exist('warnDuplicates') && warnDuplicates
			detectDuplicates(d, ret{d}, j, i, w_tmp)
		end;
	end
end



% Issue a warning if there are any duplicate synapses (same pre, post, and delay)
function detectDuplicates(d, wm, j, i, w)
	duplicates = 0;
	for n = 1:length(w)
		if wm(j(n), i(n)) ~= w(n)
			duplicates = duplicates + 1;
		end;
	end;

	if duplicates ~= 0
		msg = sprintf('Connectivity matrix contains %u duplicates for delay %u', duplicates, d);
		warning(msg);
	end;
end
