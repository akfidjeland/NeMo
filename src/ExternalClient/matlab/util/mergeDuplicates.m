% Remove duplicate synapses from connectivity matrix
%
%	mergeDuplicates(TARGETS, DELAYS, WEIGHTS)
%
% Two synapses are considered duplicates if they have the same source, target,
% and delay. Such duplicates are merged by adding their weights.
%
% Invalid synapses are assumed to be indicated by a target of 0. Thus, only the
% target matrix is returned, while the delay and weight matrices are left
% unchanged.

function [targets1, delays1, weights1] = mergeDuplicates(targets, delays, weights)

	d_max = max(max(delays));

	% Turning this into a sparse matrix will merge duplicates
	cm = sparsify(targets, delays, weights, d_max, false);

	targets1 = zeros(size(targets));
	delays1 = ones(size(delays));
	weights1 = zeros(size(weights));

	ncount = size(targets, 1);

	for pre = 1:ncount

		tr = [];
		dr = [];
		wr = [];

		for delay = 1:d_max
			m = cm{delay};
			post = find(m(:,pre) ~= 0);
			tr = [tr, post'];
			wr = [wr, m(post, pre)'];
			dr = [dr, repmat(delay, 1, length(post))];
		end;

		targets1(pre, 1:length(tr)) = tr;
		delays1(pre, 1:length(dr)) = dr;
		weights1(pre, 1:length(wr)) = wr;
	end;
end
