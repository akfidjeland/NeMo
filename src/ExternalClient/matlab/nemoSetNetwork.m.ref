function nemoSetNetwork(a, b, c, d, u, v, targets, delays, weights, plastic)

	if ~exist('plastic', 'var')
		plastic = false(size(targets));
	end;

	global NEMO_NEURONS_A;
	global NEMO_NEURONS_B;
	global NEMO_NEURONS_C;
	global NEMO_NEURONS_D;
	global NEMO_NEURONS_U;
	global NEMO_NEURONS_V;

	NEMO_NEURONS_A = a;
	NEMO_NEURONS_B = b;
	NEMO_NEURONS_C = c;
	NEMO_NEURONS_D = d;
	NEMO_NEURONS_U = u;
	NEMO_NEURONS_V = v;

	global NEMO_CM; % sparse connectivity matrix, stored by delay

	% TODO: The accumulator will take of a lot of space, which may not be
	% needed
	global NEMO_MAX_DELAY;

	NEMO_MAX_DELAY = max(max(delays));
	fprintf('Creating forward connectivity matrix...');
	NEMO_CM = sparsify(targets, delays, weights, NEMO_MAX_DELAY);
	fprintf('done\n');
	ncount = length(a);
	% TODO: which way should we orient
	% TODO: limit to size of STDP window

	global NEMO_RECENT_FIRING;
	NEMO_RECENT_FIRING = false(ncount, 64);

	fprintf('Creating reverse connectivity matrix...');
	set_reverse_d(targets, delays, plastic, ncount, NEMO_MAX_DELAY)
	fprintf('done\n');
end



% Set reverse matrix split by delay.
function set_reverse_d(targets, delays, plastic, ncount, d_max)

	global NEMO_RCM;

	global NEMO_STDP_ACC;
	global NEMO_RCM_VALID;
	global NEMO_RCM_CHANGED;

	NEMO_RCM = cell(ncount, d_max);
	NEMO_STDP_ACC = cell(ncount, d_max);
	NEMO_RCM_VALID = false(ncount, d_max);
	NEMO_RCM_CHANGED = false(ncount, d_max);

	[ncount scount] = size(targets);
	pre = repmat([1:ncount]', scount, 1);

	% Ignore all synapses marked invalid (target 0)
	% Also ignore synapses not marked as 'plastic'
	valid = find(targets ~= 0 & plastic ~= 0);

	if(isempty(valid))
		warning('nemo:info', 'No plastic synapses specified. STDP will have no effect.');
	end;

	pre_v = pre(valid);
	delays_v = int32(reshape(delays, ncount*scount, 1));
	targets_v = int32(reshape(targets, ncount*scount, 1));

	NEMO_RCM = accumarray([targets_v(valid), ...
		delays_v(valid)], ...
		pre_v, ...
		[ncount d_max], @(x) {x'});

	for n=1:ncount
		for d=1:d_max
			NEMO_STDP_ACC{n,d} = 0 * double(NEMO_RCM{n,d});
			NEMO_RCM_VALID(n,d) = ~isempty(NEMO_RCM{n,d});
		end
	end
end
