% nsEnableSTDP: enable and configure STDP
%
%   nsEnableSTDP(TAU_P, TAU_D, ALPHA_P, ALPHA_D, MAX_WEIGHT)
%
% If set before nsStart is called, the backend is configured to run with STDP.
% It will gather depression and potentiation statistics continously. To update
% the synapses using the accumulated values, call nsApplySTDP.
%
% The synapse modification is calculated based on the time difference, dt,
% between a spike arriving and a postsynaptic neuron firing, based on the
% following formula:
%
% f(dt) = ALPHA * exp(-dt / TAU)  if dt < TAU
%
% The parameters can differ depending on whether the update is potentiation
% (subscript 'P'), if the firing occurs after the spike arrival, or depression
% (subscript 'D'), if the firing occurs before the spike arrival.
%
% Only excitatory synapses are affectd by STDP. The weights of the affected
% synapses are never allowed to increase above MAX_WEIGHT, and likewise not
% allowed to go negative. Once a synapse reaches the weight 0, it never
% recovers.
%
% STDP is disabled by default. When nsEnableSTDP is called it is enabled for
% all subsequent simulations until nsDisableSTDP is called. 

% Just store the values, configuration is done in nsStart
function nsEnableSTDP(tau_p, tau_d, alpha_p, alpha_d, maxWeight)
    global NS_STDP_ACTIVE;
    global NS_STDP_TAU_P;
    global NS_STDP_TAU_D;
    global NS_STDP_ALPHA_P;
    global NS_STDP_ALPHA_D;
    global NS_STDP_MAX_WEIGHT;

    NS_STDP_ACTIVE = int32(1);
    NS_STDP_TAU_P = int32(tau_p);
    NS_STDP_TAU_D = int32(tau_d);
    NS_STDP_ALPHA_P = alpha_p;
    NS_STDP_ALPHA_D = alpha_d;
    NS_STDP_MAX_WEIGHT = maxWeight;
end
