{-# LANGUAGE TypeSynonymInstances #-}

{- Collection of synapses with the same source neuron.
 -
 - The term 'axon' is clearly used a bit loosely here, since the underlying
 - model has a rather abstracted view of connections between neurons.
 -
 - This module is best imported qualified to avoid name clashes
 -
 - Errors are reported using monadic failure, leaving the user with the choice
 - of monad to run in (Maybe, Either, and IO make sense). Functions which can
 - fail like this are given the subscript M.
 -}

module Construction.Axon (
        -- * Construct
        Axon(..),
        unconnected,
        fromList,
        -- * Query
        synapses,
        synapsesByDelay,
        size,
        maxDelay,
        targets,
        -- * Modify
        connect,
        connectMany,
        disconnect,
        disconnectM,
        replaceM,
        -- * Traverse
        withTargets,
        withSynapses,
        -- * Pretty-print
        printConnections,
        -- * Internals, exposed for testing
        present
    ) where


import Control.Monad (forM_)
import Data.List (foldl', intercalate, find, delete)
import qualified Data.Map as Map
import Data.Maybe (isJust)
import Control.Parallel.Strategies (NFData, rnf, using)

import Construction.Synapse (Synapse(..), delay, Static)
import Types (Source, Target, Delay)
import qualified Util.List as L (replace)
import qualified Util.Assocs as Assocs (mapElems)


{- Synapses are stored sorted by delay, since this is how the backend will need
 - to access them. We only need the postsynaptic index and the "payload" (which
 - varies depending on synapse type). The delay and the source neuron index is
 - stored in the collection data structure, rather than in the leaf nodes. -}

newtype Axon s = Axon {
        smap :: Map.Map Delay (AxonD s)
    } deriving (Eq)


{- All synapses with a specific delay are stored indexed by the target neruon.
 -
 - If two synapses have the same source, target, and delay, they will end up in
 - the same leaf node. It may or may not be desirable to merge such parallel
 - synapses. With straightforward static synapses, it's probably desirable to
 - merge these (to decrease synapse count, spike deliveries, and the
 - possibility of collisions in the current update on the backend), although
 - the network should produce the same result in either case. For plastic
 - synapses, however, merging synapses could affect the learning, especially,
 - if an additive scheme is used for potentiation/depression.
 -
 - The treatment of parallel synapses really ought to be specified by the user.
 -}

type AxonD s = Map.Map Target [s]


{- | Return a list of target/synapse pairs for a bundle of synapses with a
 - fixed delay -}
synapsesD = concat . map expand . Map.assocs
    where
        expand (tgt, ss) = map ((,) tgt) ss


{- | Return the number of synapses in a synapse bundle with a fixed delay -}
sizeD :: AxonD s -> Int
sizeD = Map.fold (\xs a -> a + length xs) 0


{- | Return axon with no synapses -}
unconnected :: Axon s
unconnected = Axon Map.empty


{- | Create axon from a list of synapses -}
fromList :: [Synapse s] -> Axon s
fromList = foldl' (flip connect) unconnected


{- | Return all synapses, not ordered. The source must be supplied since a
 - stand-alone synapse should contain the source, and the source is not present
 - in the data.  -}
synapses :: Source -> Axon s -> [Synapse s]
synapses src axon = concat $ map wrap $ synapsesByDelay axon
    where
        wrap (d, ss) = map (\(tgt, s) -> Synapse src tgt d s) ss


{- | Return all synapses without the source -}
strippedSynapses :: Axon s -> [(Target, Delay, s)]
strippedSynapses axon = concat $ map wrap $ synapsesByDelay axon
    where
        wrap (d, ss) = map (\(tgt, s) -> (tgt, d, s)) ss


{- | Return all synapses, ordered by delay -}
synapsesByDelay :: Axon s -> [(Delay, [(Target, s)])]
synapsesByDelay (Axon ss) = Assocs.mapElems synapsesD $ Map.toList ss


{- | Return number of synapses -}
size :: Axon s -> Int
size = sum . map sizeD . Map.elems . smap


maxDelay :: Axon s -> Delay
maxDelay (Axon ss) =
    if Map.null ss
        then 0
        else fst $ Map.findMax $ ss


{- | Return list of all targets, including duplicates -}
targets :: Axon s -> [Target]
targets (Axon ss) = concatMap targetsD $ Map.elems ss
    where
        targetsD ssD = concatMap targetsDT $ Map.assocs ssD
        targetsDT (tgt, ss) = replicate (length ss) tgt


{- | Add a synapse to axon. Duplicates are kept -}
connect = connectWith (++)


{- | Add a synapse with a specified combining function to use in case two
 - synapses have the same source, target, and delay -}
connectWith :: ([s] -> [s] -> [s]) -> Synapse s -> Axon s -> Axon s
connectWith f s (Axon ss) =
    Axon $ Map.alter (go (target s) (sdata s)) (delay s) ss
    where
        go t s Nothing = Just $ Map.singleton t [s]
        go t s (Just ss) =
           let ss' = Map.insertWith f t [s] ss in ss' `seq` Just ss'


{- | Add a group of synapses -}
connectMany :: [Synapse s] -> Axon s -> Axon s
connectMany ss axon = foldl' (flip connect) axon ss


{- | Check if synapse is part of an axon -}
present :: (Eq s) => Synapse s -> Axon s -> Bool
present s (Axon ss) = isJust found
   where
       found = find (== (sdata s)) =<< Map.lookup (target s) =<< Map.lookup (delay s) ss


{- | Remove the first matching synapse -}
disconnect :: (Eq s) => Synapse s -> Axon s -> Axon s
disconnect s a@(Axon ss) =
    -- TODO: do lookup and delete in one go
    if present s a
        then Axon $ Map.adjust (Map.adjust (delete (sdata s)) (target s)) (delay s) ss
        else a


{- | Remove the first matching synapse, reporting error in monad if no match is
 - found -}
disconnectM
    :: (Monad m, Eq s, Show s)
    => Synapse s -> Axon s -> m (Axon s)
disconnectM s axon =
    if present s axon
        then return $! disconnect s axon
        else fail $ "disconnectM: synapse not found" ++ show s


{- | Replace the *first* matching synapse, reporting error if non-existent -}
replaceM
    :: (Monad m, Show s, Eq s)
    => Synapse s -> Synapse s -> Axon s -> m (Axon s)
replaceM old new axon =
    if present old axon
        then return $ connect new $ disconnect old axon
        else fail $ "Axon.replace: failed to find synapse " ++ show old


{- | Map function over all target indices -}
withTargets :: (Target -> Target) -> Axon s -> Axon s
withTargets f (Axon ss) = Axon $ Map.map go ss
    where
        -- TODO: we should use same merging scheme as in 'connect'
        go  = Map.mapKeysWith err f
        err = error "Axon.withTargets: updated axon contains duplicate targets"


{- | Map function over all synapse data -}
withSynapses :: (s -> s) -> Axon s -> Axon s
withSynapses f (Axon ss) = Axon $ Map.map (Map.map (map f)) ss


-- TODO: make instance of Show instead
printConnections :: (Show s) => Source -> Axon s -> IO ()
printConnections src axon = do
    forM_ (synapsesByDelay axon) $ \(d, ss) -> do
        forM_ ss $ \(tgt, s) -> do
            putStrLn $ intercalate " " $ [show src, show tgt, show d, show s]


instance (Show s) => Show (Axon s) where
    {- Print one synapse per line -}
    showsPrec _ a s = showSynapses (strippedSynapses a) s
        where
            showSynapses [] = id
            showSynapses (s:ss) = shows s . showChar '\n' . showSynapses ss


instance (NFData s) => NFData (Axon s) where
    rnf (Axon ss) = rnf ss `seq` ()
