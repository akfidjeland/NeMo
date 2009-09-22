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
        synapsesUnordered,
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
        hPrintConnections,
        -- * Internals, exposed for testing
        present,
        sort,
        isSorted
    ) where


import Control.Monad (forM_)
import Control.Parallel.Strategies (NFData, rnf, using)
import Data.List (foldl', intercalate, find, delete)
import qualified Data.Map as Map
import Data.Maybe (isJust)
import System.IO (Handle, hPutStrLn)

import Construction.Synapse (Synapse(..), delay, Static, mapTarget, mapSData, changeSource)
import Types (Source, Target, Delay)
import qualified Util.List as L (replace, maxOr0)
import qualified Util.Assocs as Assocs (mapElems)


{- Synapses are stored sorted by delay, since this is how the backend will need
 - to access them. We only need the postsynaptic index and the "payload" (which
 - varies depending on synapse type). The delay and the source neuron index is
 - stored in the collection data structure, rather than in the leaf nodes. -}

data Axon s
        = Unsorted ![Synapse s]
        | Sorted { smap :: AxonS s}
    deriving (Eq)


isSorted (Unsorted _) = False
isSorted (Sorted _) = True

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

type AxonS s = Map.Map Delay (AxonD s)
type AxonD s = Map.Map Target [s]



{- | Convert from unsorted to sorted representation
 -
 - We do a reversal of the unsorted synapses to ensure that the resulting
 - Data.Map has exactly the same structure as if the axon was sorted from the
 - beginning. Using a Data.Sequence might be a better choice here.
 - -}
sort :: Axon s -> Axon s
sort (Unsorted ss) = Sorted $ foldl' (flip connectSorted) Map.empty $ reverse ss
sort axon@(Sorted _) = axon


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
unconnected = Unsorted []


{- | Create axon from a list of synapses -}
fromList :: [Synapse s] -> Axon s
fromList ss = Unsorted $ reverse ss


{- | Return all synapses, ordered by delay and target. The source must be
 - supplied since a stand-alone synapse should contain the source, and the
 - source is not necessarily present in the data.  -}
synapses :: Source -> Axon s -> [Synapse s]
synapses src axon@(Unsorted ss) = synapses src $ sort axon
synapses src axon@(Sorted _) = concat $ map wrap $ synapsesByDelay axon
    where
        wrap (d, ss) = map (\(tgt, s) -> Synapse src tgt d s) ss


{- | Return all synapses.
 -
 - This function differs from *synapses* in that the order of the synapses may
 - differ depending on how the axon has been modified previously. If the neuron
 - is currently unsorted the neurons will be returned in the same order in
 - which they were inserted. This method will be cheaper for unsorted axons.-}
synapsesUnordered src (Unsorted ss) = reverse $ fmap (changeSource src) ss
synapsesUnordered src axon@(Sorted _) = synapses src axon


{- | Return all synapses without the source -}
strippedSynapses :: Axon s -> [(Target, Delay, s)]
strippedSynapses (Unsorted ss) = fmap strip ss
    where
        strip s = (target s, delay s, sdata s)
strippedSynapses axon@(Sorted _) = concat $ map wrap $ synapsesByDelay axon
    where
        wrap (d, ss) = map (\(tgt, s) -> (tgt, d, s)) ss


{- | Return all synapses, ordered by delay -}
synapsesByDelay :: Axon s -> [(Delay, [(Target, s)])]
synapsesByDelay axon =
    case axon of
        (Unsorted _)-> synapsesByDelay $ sort axon
        (Sorted ss) -> Assocs.mapElems synapsesD $ Map.toList ss


{- | Return number of synapses -}
size :: Axon s -> Int
-- TODO: perhaps keep track of length when building up axon
size (Unsorted ss) = length ss
size (Sorted ss) = sum $ map sizeD $ Map.elems ss


maxDelay :: Axon s -> Delay
maxDelay (Unsorted ss) = L.maxOr0 $ fmap delay ss
maxDelay (Sorted ss) =
    if Map.null ss
        then 0
        else fst $ Map.findMax $ ss


{- | Return list of all targets, including duplicates -}
targets :: Axon s -> [Target]
targets (Unsorted ss) = fmap target ss
targets (Sorted ss) = concatMap targetsD $ Map.elems ss
    where
        targetsD ssD = concatMap targetsDT $ Map.assocs ssD
        targetsDT (tgt, ss) = replicate (length ss) tgt


{- | Add a synapse to axon. Duplicates are kept -}
connect :: Synapse s -> Axon s -> Axon s
connect s (Unsorted ss) = Unsorted (s:ss)
connect s axon@(Sorted ss) = Sorted $ connectSorted s ss


connectSorted :: Synapse s -> AxonS s -> AxonS s
connectSorted = connectSortedWith (++)


{- | Add a synapse with a specified combining function to use in case two
 - synapses have the same source, target, and delay -}
connectSortedWith :: ([s] -> [s] -> [s]) -> Synapse s -> AxonS s -> AxonS s
connectSortedWith f s ss = Map.alter (go (target s) (sdata s)) (delay s) ss
    where
        go t s Nothing = Just $ Map.singleton t [s]
        go t s (Just ss) =
           let ss' = Map.insertWith f t [s] ss in ss' `seq` Just ss'


{- | Add a group of synapses -}
connectMany :: [Synapse s] -> Axon s -> Axon s
-- connectMany ss' (Unsorted ss) = Unsorted $ (reverse ss') ++ ss
connectMany ss' (Unsorted ss) = Unsorted $ (reverse ss') ++ ss
connectMany ss' (Sorted ss) = Sorted $ foldl' (flip connectSorted) ss ss'


{- | Check if synapse is part of an axon -}
present :: (Eq s) => Synapse s -> AxonS s -> Bool
present s ss = isJust found
   where
       found = find (== (sdata s)) =<< Map.lookup (target s) =<< Map.lookup (delay s) ss



{- | Remove the first matching synapse -}
disconnect :: (Eq s) => Synapse s -> Axon s -> Axon s
disconnect s axon =
    case axon of
        (Unsorted _) -> disconnect s $ sort axon
        (Sorted ss) ->
            if present s ss
                -- TODO: do lookup and delete in one go
                then Sorted $ Map.adjust (Map.adjust (delete (sdata s)) (target s)) (delay s) ss
                else axon


{- | Remove the first matching synapse, reporting error in monad if no match is
 - found -}
disconnectM
    :: (Monad m, Eq s, Show s)
    => Synapse s -> Axon s -> m (Axon s)
disconnectM s axon =
    case axon of
        (Unsorted _) -> disconnectM s $ sort axon
        -- TODO: avoid checking for presence twice: factor out the raw
        -- disconnection code
        (Sorted ss) -> if present s ss
            then return $! disconnect s axon
            else fail $ "disconnectM: synapse not found" ++ show s


{- | Replace the *first* matching synapse, reporting error if non-existent -}
replaceM
    :: (Monad m, Show s, Eq s)
    => Synapse s -> Synapse s -> Axon s -> m (Axon s)
replaceM old new axon =
    case axon of
        (Unsorted _) -> replaceM old new $ sort axon
        (Sorted ss) ->
            if present old ss
                then return $ connect new $ disconnect old axon
                else fail $ "Axon.replace: failed to find synapse " ++ show old


{- | Map function over all target indices -}
withTargets :: (Target -> Target) -> Axon s -> Axon s
withTargets f (Unsorted ss) = Unsorted $ fmap (mapTarget f) ss
withTargets f (Sorted ss) = Sorted $ Map.map go ss
    where
        -- TODO: we should use same merging scheme as in 'connect'
        go  = Map.mapKeysWith err f
        err = error "Axon.withTargets: updated axon contains duplicate targets"


{- | Map function over all synapse data -}
withSynapses :: (s -> s) -> Axon s -> Axon s
withSynapses f (Unsorted ss) = Unsorted $ fmap (mapSData f) ss
withSynapses f (Sorted ss) = Sorted $ Map.map (Map.map (map f)) ss


-- TODO: make instance of Show instead
hPrintConnections :: (Show s) => Handle -> Source -> Axon s -> IO ()
hPrintConnections hdl src axon = do
    forM_ (synapsesByDelay axon) $ \(d, ss) -> do
        forM_ ss $ \(tgt, s) -> do
            hPutStrLn hdl $ (show src) ++ " -> " ++ (intercalate " " $ [show tgt, show d, show s])


instance (Show s) => Show (Axon s) where
    {- Print one synapse per line -}
    showsPrec _ a s = showSynapses (strippedSynapses a) s
        where
            showSynapses [] = id
            showSynapses (s:ss) = shows s . showChar '\n' . showSynapses ss

instance (NFData s) => NFData (Axon s) where
    rnf (Unsorted ss) = rnf ss `seq` ()
    rnf (Sorted ss) = rnf ss `seq` ()
