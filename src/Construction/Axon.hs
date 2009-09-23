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
        withWeights,
        -- * Pretty-print
        hPrintConnections,
        -- * Internals, exposed for testing
        present,
        sort,
        strip
    ) where


import Control.Monad (forM_)
import Control.Parallel.Strategies (NFData, rnf, using)
import Data.Foldable (foldl', toList)
import Data.List (intercalate, find, delete)
-- TODO: remove qualification here
import qualified Data.Sequence as Seq
import qualified Data.Map as Map
import Data.Maybe (isJust)
import System.IO (Handle, hPutStrLn)

import Construction.Synapse (Synapse(..), delay, Static, changeSource)
import Types (Source, Target, Delay, Current)
import qualified Util.List as L (replace, maxOr0)
import qualified Util.Assocs as Assocs (mapElems)



{- Synapses are stored using two different schemes. The Unsorted axon has cheap
 - insertion (in terms of space at least), but more expensive query and
 - modification, while the Sorted axon has cheaper query and modification. An
 - axon always start out being unsorted and is sorted only when needed -}

data Axon s
        = Unsorted !(Seq.Seq (Stripped s)) -- first insertion stored leftmost
        | Sorted (AxonS s)
    deriving (Eq)


isSorted (Unsorted _) = False
isSorted (Sorted _) = True


{- Internally, we don't need the source neuron, so we store a stripped synapse
 - to save space. -}

-- TODO: perhaps move this to Construction.Synapse
data Stripped s = Stripped {
        sTarget :: {-# UNPACK #-} !Target,
        sDelay  :: {-# UNPACK #-} !Delay,
        sWeight :: {-# UNPACK #-} !Current,
        -- TODO: may want variable payload, but with specialisation for just a double
        sAux    :: {-# UNPACK #-} !s
    } deriving (Eq, Show, Ord)


strip :: Synapse s -> Stripped s
strip s = Stripped (target s) (delay s) (weight s) (sdata s)

unstrip :: Source -> Stripped s -> Synapse s
unstrip src (Stripped t d w a) = Synapse src t d w a

mapTarget f (Stripped t d w a) = Stripped (f t) d w a
mapWeight f (Stripped t d w a) = Stripped t d (f w) a



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
-- TODO: make sure we use unboxed tuple here
type AxonD s = Map.Map Target [Leaf s]

-- data Leaf = Leaf {-# UNPACK #-} !Current {-# UNPACK #-} !s
type Leaf s = (Current, s)


withLeafWeight :: (Current -> Current) -> Leaf s -> Leaf s
withLeafWeight f (w, s) = (f w, s)



{- | Convert from unsorted to sorted representation
 -
 - We do a reversal of the unsorted synapses to ensure that the resulting
 - Data.Map has exactly the same structure as if the axon was sorted from the
 - beginning. Using a Data.Sequence might be a better choice here.
 - -}
sort :: Axon s -> Axon s
{-# SPECIALIZE sort :: Axon () -> Axon () #-}
sort (Unsorted ss) = Sorted $ foldl' (flip connectSorted) Map.empty ss
sort axon@(Sorted _) = axon


{- | Return a list of target/synapse pairs for a bundle of synapses with a
 - fixed delay -}
synapsesD :: AxonD s -> [(Target, Current, s)]
synapsesD = concat . map expand . Map.assocs
    where
        -- expand (tgt, ss) = map ((,) tgt) ss
        expand (tgt, ss) = map (\(w, pl) -> (tgt, w, pl)) ss


{- | Return the number of synapses in a synapse bundle with a fixed delay -}
sizeD :: AxonD s -> Int
sizeD = Map.fold (\xs a -> a + length xs) 0


{- | Return axon with no synapses -}
unconnected :: Axon s
unconnected = Unsorted Seq.empty


{- | Create axon from a list of synapses -}
fromList :: [Synapse s] -> Axon s
fromList ss = Unsorted $ Seq.fromList $ map strip ss


{- | Return all synapses, ordered by delay and target. The source must be
 - supplied since a stand-alone synapse should contain the source, and the
 - source is not necessarily present in the data.  -}
synapses :: Source -> Axon s -> [Synapse s]
synapses src axon@(Unsorted ss) = synapses src $ sort axon
synapses src axon@(Sorted _) = concat $ map wrap $ synapsesByDelay axon
    where
        wrap (d, ss) = map (\(tgt, w, s) -> Synapse src tgt d w s) ss


{- | Return all synapses.
 -
 - This function differs from 'synapses' in that the order of the synapses may
 - differ depending on how the axon has been modified previously. If the neuron
 - is currently unsorted the neurons will be returned in the same order in
 - which they were inserted. This method will be cheaper for unsorted axons.-}
synapsesUnordered src (Unsorted ss) = toList $ fmap (unstrip src) ss
synapsesUnordered src axon@(Sorted _) = synapses src axon


{- | Return all synapses without the source -}
strippedSynapses :: Axon s -> [(Target, Delay, Current, s)]
strippedSynapses (Unsorted ss) = toList $ fmap strip ss
    where
        strip (Stripped t d w a) = (t, d, w, a)
strippedSynapses axon@(Sorted _) = concat $ map wrap $ synapsesByDelay axon
    where
        wrap (d, ss) = map (\(tgt, w, s) -> (tgt, d, w, s)) ss


{- | Return all synapses, ordered by delay -}
synapsesByDelay :: Axon s -> [(Delay, [(Target, Current, s)])]
synapsesByDelay axon =
    case axon of
        (Unsorted _)-> synapsesByDelay $ sort axon
        (Sorted ss) -> Assocs.mapElems synapsesD $ Map.toList ss


{- | Return number of synapses -}
size :: Axon s -> Int
size (Unsorted ss) = Seq.length ss
size (Sorted ss) = sum $ map sizeD $ Map.elems ss


maxDelay :: Axon s -> Delay
maxDelay (Unsorted ss) = L.maxOr0 $ toList $ fmap sDelay ss
maxDelay (Sorted ss) =
    if Map.null ss
        then 0
        else fst $ Map.findMax $ ss


{- | Return list of all targets, including duplicates -}
targets :: Axon s -> [Target]
targets (Unsorted ss) = toList $ fmap sTarget ss
targets (Sorted ss) = concatMap targetsD $ Map.elems ss
    where
        targetsD ssD = concatMap targetsDT $ Map.assocs ssD
        targetsDT (tgt, ss) = replicate (length ss) tgt


{- | Add a synapse to axon. Duplicates are kept -}
connect :: Synapse s -> Axon s -> Axon s
{-# SPECIALIZE connect :: Synapse () -> Axon () -> Axon () #-}
connect s (Unsorted ss) = Unsorted $ (Seq.|>) ss $! strip s
connect s axon@(Sorted ss) = Sorted $ connectSorted (strip s) ss


connectSorted :: Stripped s -> AxonS s -> AxonS s
{-# SPECIALIZE connectSorted :: Stripped () -> AxonS () -> AxonS () #-}
connectSorted = connectSortedWith (++)


{- | Add a synapse with a specified combining function to use in case two
 - synapses have the same source, target, and delay -}
connectSortedWith :: ([Leaf s] -> [Leaf s] -> [Leaf s]) -> Stripped s -> AxonS s -> AxonS s
connectSortedWith f s ss = Map.alter (go (sTarget s) (sWeight s) (sAux s)) (sDelay s) ss
    where
        go t w s Nothing = Just $ Map.singleton t [(w, s)]
        go t w s (Just ss) =
           let ss' = Map.insertWith f t [(w, s)] ss in ss' `seq` Just ss'


{- | Add a group of synapses -}
connectMany :: [Synapse s] -> Axon s -> Axon s
connectMany ss' (Unsorted ss) = Unsorted $ ss Seq.>< (Seq.fromList $ map strip ss')
connectMany ss' (Sorted ss) = Sorted $ foldl' (flip connectSorted) ss $ map strip ss'


{- | Check if synapse is part of an axon -}
present :: (Eq s) => Synapse s -> AxonS s -> Bool
present s ss = isJust found
   where
       found = find eq =<< Map.lookup (target s) =<< Map.lookup (delay s) ss
       eq (w, pl) = w == weight s && pl == sdata s



{- | Remove the first matching synapse -}
disconnect :: (Eq s) => Synapse s -> Axon s -> Axon s
disconnect s axon =
    case axon of
        (Unsorted _) -> disconnect s $ sort axon
        (Sorted ss) ->
            if present s ss
                -- TODO: do lookup and delete in one go
                then Sorted $ Map.adjust (Map.adjust (delete ((weight s, sdata s))) (target s)) (delay s) ss
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


{- | Map function over all weights -}
withWeights :: (Current -> Current) -> Axon s -> Axon s
withWeights f (Unsorted ss) = Unsorted $ fmap (mapWeight f) ss
withWeights f (Sorted ss) = Sorted $ Map.map (Map.map (map $ withLeafWeight f)) ss


-- TODO: make instance of Show instead
hPrintConnections :: (Show s) => Handle -> Source -> Axon s -> IO ()
hPrintConnections hdl src axon = do
    forM_ (synapsesByDelay axon) $ \(d, ss) -> do
        forM_ ss $ \(tgt, w, s) -> do
            hPutStrLn hdl $ (show src) ++ " -> " ++ (intercalate " " $ [show tgt, show d, show w, show s])


instance (Show s) => Show (Axon s) where
    {- Print one synapse per line -}
    showsPrec _ a s = showSynapses (strippedSynapses a) s
        where
            showSynapses [] = id
            showSynapses (s:ss) = shows s . showChar '\n' . showSynapses ss

instance (NFData s) => NFData (Stripped s) where
    rnf (Stripped t d w a) = rnf t `seq` rnf d `seq` rnf w `seq` rnf a `seq` ()

instance (NFData s) => NFData (Axon s) where
    rnf (Unsorted ss) = (rnf $! toList ss) `seq` ()
    rnf (Sorted ss) = rnf ss `seq` ()
