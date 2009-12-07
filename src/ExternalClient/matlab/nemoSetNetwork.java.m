function nemoSetNetwork(a, b, c, d, u, v, targets, delays, weights, plastic)

	global NEMO_INSTANCE;
	if exist('NEMO_INSTANCE') ~= 1
		error('No running instance of nemo found');
	end

	[ncount, scount] = size(targets);

	% All postsynaptic indices should be in the network
	checkTargets(targets, size(a,1));

	sd = transpose(int16(delays));
	checkDelays(sd);
	sd = reshape(sd, ncount*scount, 1);

	% adjust indices by one as host expects 0-based array indexing
	st = reshape(transpose(int32(targets-1)), ncount*scount, 1);
	sw = reshape(transpose(weights), ncount*scount, 1);
	ps = reshape(transpose(logical(plastic)), ncount*scount, 1);

	NEMO_INSTANCE.setNetwork(a, b, c, d, u, v, st, sd, sw, ps);
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
