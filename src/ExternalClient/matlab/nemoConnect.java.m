function nemoConnect(host, port)

	nemo_path = getenv('NEMO_PATH');
	if isempty(nemo_path)
		error 'The environment variable NEMO_PATH is not set'
	end
	
	javaaddpath(fullfile(nemo_path, 'nemo.jar'));

	% TODO: get default port from somewhere else, use m4 to replace value
	if nargin < 2, port = 56100; end
	if nargin < 1, host = 'localhost'; end

	global NEMO_INSTANCE;
	
	% TODO: check for existing connection
	NEMO_INSTANCE = nemo.MatlabClient(host, port);
end
