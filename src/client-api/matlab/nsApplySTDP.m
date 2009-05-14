% nsApplySTDP: update synapses
%
%   nsApplySTDP(R)
%
% Update synapses using potentiation and depression accumulated so far. The
% synapses update is multiplied with the reward signal R.
%
% The potentiation and depression accumulators are cleared. 

% We only apply STDP on the next sync call to the host
function nsApplySTDP(r)
   
    global NS_STDP_APPLY;
    global NS_STDP_REWARD;
    global NS_STDP_ACTIVE;

    if isa(NS_STDP_ACTIVE, 'int32') && NS_STDP_ACTIVE ~= 0
        NS_STDP_APPLY = int32(1);
        NS_STDP_REWARD = r;
    else
        NS_STDP_APPLY = int32(0);
        warning 'Simulation not configured for STDP, synapses not updated';
    end;
end
