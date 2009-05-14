module Examples.Smallworld (smallworld, smallworldOrig) where
import Construction

-- Smallworld topology network

-- Scaling factor (applied when generating synapses)
-- f = 30.0
-- f = 34.0


-- Excitatory synapse with user-specified delay
exSd delay f = mkRSynapse (between 0.0 (0.7*f)) delay


-- Excitatory synapse with random delay
exS maxDelay f = exSd (between 1 maxDelay) f


-- Randomised excitatory neuron
exN useThalamic r = mkNeuron2 0.02 b (v + 15*r^2) (8.0-6.0*r^2) u v thalamic
    where
        b = 0.2
        u = b * v
        v = -65.0
        thalamic = if useThalamic then mkThalamic 5.0 r else Nothing

-- Inhibitory synapse
inS f = mkRSynapse (between 0.0 (-2.0*f)) (fixed 1)


-- Inhibitory neuron
inN useThalamic r = mkNeuron2 (0.02 + 0.08*r) b c 2.0 u v thalamic
    where
        b = 0.25 - 0.05 * r
        c = v
        u = b * v
        v = -65.0
        thalamic = if useThalamic then mkThalamic 2.0 r else Nothing


-- Excitatory cluster
exC sz nss maxDelay f useTh = clusterN (replicate sz (randomised (exN useTh)))
         [ connect every (nonself |> random nss) (exS maxDelay f)]
         -- [ connect every (ignorePre |> random nss) (exS maxDelay f)]


-- Inhibitory cluster of n neurons
inC sz useTh = clusterN (replicate sz (randomised (inN useTh))) []


-- Combined cluster with both excitatory and inhibitory neurons
-- 5 excitatory connections from each excitatory to 5 random inhibitory.
-- 20 inhibitory connetions from each inhibitory to random excitatory within cluster.
cc sz synapseCount maxDelay f useTh = cluster [exC exSz exSCount maxDelay f useTh, inC inSz useTh]
        -- TODO: use either subnet names or neuron properties
        [ connect (nth 1) (ccaRoot |> random inSCount) (exSd (fixed 1) f)
        , connect (nth 2) (ccaRoot |> random synapseCount) (inS f)]
    where
        exSz = sz * 8 `div` 10
        inSz = sz - exSz
        inSCount = synapseCount `div` 5
        exSCount = synapseCount - inSCount
        -- alternative specification:
        -- [ connect (everySynapse excitatory) (ccaRoot |> random 4) (exSd (fixed 1))
        -- , connect (everySynapse inhibitory)  (ccaRoot |> random 20) inS ]


-- fully parameterised smallworld network
smallworld clusterCount clusterSz synapseCount p maxDelay f useThalamicInput =
    if clusterCount == 1
        then cc clusterSz synapseCount maxDelay f useThalamicInput
        else
            cluster (replicate clusterCount $ cc clusterSz synapseCount maxDelay f useThalamicInput) rewiring
    where
        rewiring = if p <= 0.0
            then []
            else [ reconnect
                (synapses |> only excitatory |> withProb p)
                -- TODO: perhaps use multiple topologies for selection, e.g.
                -- maintain topologies of excitatory and inhibitory populations
                (presynaptic |> ccaRoot |> everySynapse excitatory |> oneof) ]


-- original network configuration
smallworldOrig = smallworld 10 100 20 0.01 20 30 False
