% nemoApplySTDP: update synapses
%
%   nemoApplySTDP(R)
%
% Update synapses using potentiation and depression accumulated so far. The
% synapses update is multiplied with the reward signal R.
%
% The potentiation and depression accumulators are cleared. 

% We only apply STDP on the next sync call to the host
function nemoApplySTDP(r)
   
    global NEMO_STDP_APPLY;
    global NEMO_STDP_REWARD;
    global NEMO_STDP_ACTIVE;

    if isa(NEMO_STDP_ACTIVE, 'int32') && NEMO_STDP_ACTIVE ~= 0
        NEMO_STDP_APPLY = int32(1);
        NEMO_STDP_REWARD = r;
    else
        NEMO_STDP_APPLY = int32(0);
        warning 'Simulation not configured for STDP, synapses not updated';
    end;
end
