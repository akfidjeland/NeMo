{-# LANGUAGE TypeSynonymInstances #-}

{- | Collection of neurons -}

module Construction.Neurons (
        -- * Construction
        Neurons(..),
        empty,
        fromList,
        union,
        -- * Query
        synapsesOf,
        size,
        isEmpty,
        indices,
        neurons,
        idxBounds,
        synapses,
        weightMatrix,
        toList,
        synapseCount,
        maxSynapsesPerNeuron,
        maxDelay,
        -- * Traversal
        withSynapses,
        -- * Modification
        addNeuron,
        addNeuronGroup,
        addSynapse,
        addSynapses,
        addSynapseAssocs,
        deleteSynapse,
        updateSynapses,
        withTerminals,
        -- * Printing
        printConnections,
        printNeurons
    ) where

import Control.Parallel.Strategies (NFData, rnf)
import qualified Data.Map as Map
import Data.Maybe (fromJust)

import qualified Construction.Neuron as Neuron
import Construction.Synapse (Synapse, source)
import Types

newtype Neurons n s = Neurons {
        ndata :: (Map.Map Idx (Neuron.Neuron n s))
    } deriving (Eq)


-------------------------------------------------------------------------------
-- Construction
-------------------------------------------------------------------------------

empty = Neurons $ Map.empty


fromList :: [(Idx, Neuron.Neuron n s)] -> Neurons n s
fromList = Neurons . Map.fromList


union :: (Show n, Show s) => [Neurons n s] -> Neurons n s
union ns = Neurons $ foldl f Map.empty $ map ndata ns
    where
        f = Map.unionWithKey $ duplicateKeyError "Neurons(union)"


duplicateKeyError :: (Show n) => String -> Idx -> n -> n -> n
duplicateKeyError msg key n1 n2 =
    error $! msg ++ ": duplicate key (" ++ show key ++ ") for neurons:\n" ++
        show n1 ++ "\n" ++ show n2


-------------------------------------------------------------------------------
-- Query
-------------------------------------------------------------------------------

{- | Return synapses of a specific neuron, unordered -}
synapsesOf :: Neurons n s -> Idx -> [Synapse s]
synapsesOf (Neurons ns) ix = maybe [] (Neuron.synapses ix) $ Map.lookup ix ns


{- | Return number of neurons -}
size :: Neurons n s -> Int
size = Map.size . ndata


isEmpty :: Neurons n s -> Bool
isEmpty = Map.null . ndata


{- | Return number of synapses -}
synapseCount :: Neurons n s -> Int
synapseCount (Neurons ns) = Map.fold ((+) . Neuron.synapseCount) 0 ns


{- | Return max number of synapses per neuron -}
maxSynapsesPerNeuron :: Neurons n s -> Int
maxSynapsesPerNeuron = Map.fold (max) 0 . Map.map Neuron.synapseCount . ndata


{- | Return list of all neuron indices -}
indices :: Neurons n s -> [Idx]
indices = Map.keys . ndata


{- | Return maximum and minimum neuron indices -}
idxBounds :: Neurons n s -> (Idx, Idx)
idxBounds (Neurons ns) = (mn, mx)
    where
        (mn, _) = Map.findMin ns
        (mx, _) = Map.findMax ns


{- | Return synapses ordered by source and delay -}
synapses :: Neurons n s -> [(Idx, [(Delay, [(Idx, s)])])]
synapses = map (\(i, n) -> (i, Neuron.synapsesByDelay n)) . toList


{- | Return synapses organised by source only -}
weightMatrix :: Neurons n s -> Map.Map Idx [Synapse s]
weightMatrix = Map.mapWithKey Neuron.synapses . ndata


{- | Return network as list of index-neuron pairs -}
toList :: Neurons n s -> [(Idx, Neuron.Neuron n s)]
toList = Map.toList . ndata


{- | Return the list of neurons -}
neurons :: Neurons n s -> [Neuron.Neuron n s]
neurons = Map.elems . ndata


{- | Return maximum delay in network -}
-- TODO: maintain max as we build network?
maxDelay :: Neurons n s -> Delay
maxDelay (Neurons ns) = Map.fold go 0 ns
    where
        go n d = max (Neuron.maxDelay n) d

-------------------------------------------------------------------------------
-- Traversal
-------------------------------------------------------------------------------


withSynapses :: (s -> s) -> Neurons n s -> Neurons n s
withSynapses f (Neurons ns) = Neurons $ Map.map (Neuron.withSynapses f) ns


-------------------------------------------------------------------------------
-- Modification
-------------------------------------------------------------------------------

-- TODO: monadic error handling
withNeuron
    :: (Neuron.Neuron n s -> Neuron.Neuron n s)
    -> Idx
    -> Neurons n s
    -> Neurons n s
withNeuron f idx (Neurons ns) =
    if Map.member idx ns
        then Neurons $ Map.adjust f idx ns
        else error $! "withNeuron" ++ ": invalid neuron index (" ++ show idx ++ ")"


addNeuron :: Idx -> Neuron.Neuron n s -> Neurons n s -> Neurons n s
addNeuron idx n (Neurons ns) = Neurons $ Map.insertWithKey collision idx n ns
    where
        collision idx _ _ = error $ "duplicate neuron index: " ++ show idx


addNeuronGroup
    :: (Show n, Show s)
    => [(Idx, Neuron.Neuron n s)]
    -> Neurons n s
    -> Neurons n s
addNeuronGroup ns' (Neurons ns) = Neurons $ union ns $ Map.fromList ns'
    where
        union = Map.unionWithKey $ duplicateKeyError "Neurons(mapUnion)"


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
        f (s, s') ns =  updateNeuronSynapse (source s) s s' ns


{- | Map function over all terminals (source and target) of all synapses -}
withTerminals :: (Idx -> Idx) -> Neurons n s -> Neurons n s
withTerminals f (Neurons ns) =
    Neurons $ Map.map (Neuron.withTargets f) $ Map.mapKeys f ns


-------------------------------------------------------------------------------
-- Various
-------------------------------------------------------------------------------

instance (NFData n, NFData s) => NFData (Neurons n s) where
    rnf (Neurons ns) = rnf ns

-------------------------------------------------------------------------------
-- Printing
-------------------------------------------------------------------------------


instance (Show n, Show s) => Show (Neurons n s) where
    -- create a list of all neurons, one line per neuron
    -- TODO: show synapses as well
    showsPrec _ (Neurons ns) s = showNeuronList (Map.toList ns) s

showNeuronList [] = id
showNeuronList ((idx, n):ns) =
    shows idx . showChar ':' . shows n . showChar '\n' . showNeuronList ns


{- | Print synapses, one line per synapse -}
printConnections :: (Show s) => Neurons n s -> IO ()
printConnections (Neurons ns) =
    mapM_ (uncurry Neuron.printConnections) $ Map.assocs ns


{- | Print neurons, one line per neuron -}
printNeurons :: (Show n, Show s) => Neurons n s -> IO ()
printNeurons (Neurons ns) = mapM_ f $ Map.assocs ns
    where f (idx, n) = putStrLn $ show idx ++ " " ++ show n

