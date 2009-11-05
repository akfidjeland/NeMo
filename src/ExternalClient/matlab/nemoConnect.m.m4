function nemoConnect(host, port)

	% TODO: get default port from somewhere else, use m4 to replace value
	if nargin < 2, port = 56100; end
	if nargin < 1, host = 'localhost'; end

	nemo_mex(mex_connect, host, int32(port));
end
