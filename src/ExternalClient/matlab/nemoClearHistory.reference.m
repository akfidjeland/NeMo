function nemoClearHistory

	global NEMO_RECENT_FIRING;

	if isempty(NEMO_RECENT_FIRING)
		error 'Firing history needs to be initialised before it can be cleared';
	end

	NEMO_RECENT_FIRING(:) = 0;
end
