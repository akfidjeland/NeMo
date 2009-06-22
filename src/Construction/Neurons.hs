{- | Collection of neurons -}

module Construction.Neurons (
        -- * Construction
        Neurons,
        empty,
        -- * Query
        synapsesOf,
        size,
        indices,
        idxBounds,
        synapses,
        synapseCount,
        maxDelay,
        -- * Modification
        addSynapse,
        addSynapses,
        addSynapseAssocs,
        deleteSynapse,
        updateSynapses,
        withTerminals
    ) where

import qualified Data.Map as Map
import Data.Maybe (fromJust)

import qualified Construction.Neuron as Neuron
import Construction.Synapse (Synapse, source)
import Types

type Neurons n s = Map.Map Idx (Neuron.Neuron n s)


empty = Map.empty

-------------------------------------------------------------------------------
-- Query
-------------------------------------------------------------------------------

{- | Return synapses of a specific neuron, unordered -}
synapsesOf :: Neurons n s -> Idx -> [Synapse s]
synapsesOf ns ix = maybe [] (Neuron.synapses ix) $ Map.lookup ix ns


{- | Return number of neurons -}
size :: Neurons n s -> Int
size = Map.size


{- | Return number of synapses -}
synapseCount :: Neurons n s -> Int
synapseCount = Map.fold ((+) . Neuron.synapseCount) 0


{- | Return list of all neuron indices -}
indices :: Neurons n s -> [Idx]
indices = Map.keys


{- | Return maximum and minimum neuron indices -}
idxBounds :: Neurons n s -> (Idx, Idx)
idxBounds ns = (mn, mx)
    where
        (mn, _) = Map.findMin ns
        (mx, _) = Map.findMax ns


{- | Return synapses orderd by source and delay -}
synapses :: Neurons n s -> [(Idx, [(Delay, [(Idx, s)])])]
synapses = map (\(i, n) -> (i, Neuron.synapsesByDelay n)) . Map.assocs


{- | Return maximum delay in network -}
-- TODO: maintain max as we build network?
maxDelay :: Neurons n s -> Delay
maxDelay ns = Map.fold go 0 ns
    where
        go n d = max (Neuron.maxDelay n) d

-------------------------------------------------------------------------------
-- Modification
-------------------------------------------------------------------------------

-- TODO: monadic error handling
withNeuron
    :: (Neuron.Neuron n s -> Neuron.Neuron n s)
    -> Idx
    -> Neurons n s
    -> Neurons n s
withNeuron f idx ns =
    if Map.member idx ns
        then Map.adjust f idx ns
        else error $! "withNeuron" ++ ": invalid neuron index (" ++ show idx ++ ")"


-- TODO: monadic error handling
addSynapse :: Idx -> Synapse s -> Neurons n s -> Neurons n s
addSynapse idx s ns = withNeuron (Neuron.connect s) idx ns


-- TODO: modify argument order
addSynapses :: Idx -> [Synapse s] -> Neurons n s -> Neurons n s
addSynapses idx ss ns = withNeuron (Neuron.connectMany ss) idx ns


addSynapseAssocs :: [(Idx, [Synapse s])] -> Neurons n s -> Neurons n s
addSynapseAssocs new ns = foldr (uncurry addSynapses) ns new


deleteSynapse :: (Eq s) => Idx -> Synapse s -> Neurons n s -> Neurons n s
deleteSynapse idx s ns = withNeuron (Neuron.disconnect s) idx ns


-- TODO: propagate errors from replaceSynapse
updateNeuronSynapse
    :: (Show s, Eq s)
    => Idx -> Synapse s -> Synapse s -> Neurons n s -> Neurons n s
updateNeuronSynapse idx old new ns =
    withNeuron (fromJust . Neuron.replaceSynapse old new) idx ns



{- | update several synapses in a network using a replacement list -}
updateSynapses
    :: (Show s, Eq s)
    => [(Synapse s, Synapse s)]  -- ^ replacement pair: (old, new)
    -> Neurons n s
    -> Neurons n s
-- TODO: use fold' instead?
updateSynapses diff ns = foldr f ns diff
    where
        -- f (s, s') ns =  updateNeuronSynapse (pre s) s s' ns
        f (s, s') ns =  updateNeuronSynapse (source s) s s' ns


{- | Map function over all terminals (source and target) of all synapses -}
withTerminals :: (Idx -> Idx) -> Neurons n s -> Neurons n s
withTerminals f ns = Map.map (Neuron.withTargets f) $ Map.mapKeys f ns
