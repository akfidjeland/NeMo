function nemoSetNetwork(a, b, c, d, u, v, targets, delays, weights)

	% All postsynaptic indices should be in the network
	checkTargets(targets, size(a,1));

	sd = transpose(int32(delays));
	checkDelays(sd);

	% adjust indices by one as host expects 0-based array indexing
	st = transpose(int32(targets-1));
	sw = transpose(weights);

	nemo_mex(mex_setNetwork, a, b, c, d, u, v, st, sd, sw);
end


% Check whether postsynaptic indices are out-of-bounds 
function checkTargets(targets, maxIdx)

	if ~all(all(targets >= 0))
		oob = targets(find(targets < 0));
		oob(1:min(10,size(oob,1)))
		error('Postsynaptic index matrix contains out-of-bounds members (too low). The first 10 are shown above')
	end

	if ~all(all(targets <= maxIdx)) 
		oob = targets(find(targets > maxIdx));
		oob(1:min(10,size(oob,1)))
		error('Postsynaptic index matrix contains out-of-bounds members (too high). The first 10 are shown above')
	end
end



% Check whether all delays are positive and within max
function checkDelays(delays)
	if ~all(all(delays >= 1))
		oob = delays(find(delays < 1));
		oob(1:min(10,size(oob,1)))
		error('Delay matrix contains out-of-bounds members (<1). The first 10 are shown above')
	end
	if ~all(all(delays < 32))
		oob = delays(find(delays >= 32));
		oob(1:min(10,size(oob,1)))
		error('Delay matrix contains out-of-bounds members (>=32). The first 10 are shown above')
	end
end
