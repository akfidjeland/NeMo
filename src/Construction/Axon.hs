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
        terminals,
        terminalsUnordered,
        terminalsByDelay,
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
    ) where


import Control.Monad (forM_)
import Control.Parallel.Strategies (NFData, rnf)
import Data.Foldable (foldl', toList)
import Data.List (intercalate, find, delete)
import qualified Data.Sequence as Seq
import qualified Data.Map as Map
import Data.Maybe (isJust)
import System.IO (Handle, hPutStrLn)

import Construction.Synapse (AxonTerminal(..),
        delay, target, withTarget, weight, withWeight, Static)
import Types (Source, Target, Delay, Weight)
import qualified Util.List as L (replace, maxOr0)
import qualified Util.Assocs as Assocs (mapElems)


type Terminal = AxonTerminal

{- Synapses are stored using two different schemes. The Unsorted axon has cheap
 - insertion (in terms of space at least), but more expensive query and
 - modification, while the Sorted axon has cheaper query and modification. An
 - axon always start out being unsorted and is sorted only when needed -}

data Axon s
        = Unsorted !(Seq.Seq (Terminal s)) -- first insertion stored leftmost
        | Sorted (AxonS s)
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
-- TODO: make sure we use unboxed tuple here
type AxonD s = Map.Map Target [Leaf s]

-- data Leaf = Leaf {-# UNPACK #-} !Weight {-# UNPACK #-} !s
type Leaf s = (Weight, s)


withLeafWeight :: (Weight -> Weight) -> Leaf s -> Leaf s
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
synapsesD :: AxonD s -> [(Target, Weight, s)]
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
fromList :: [Terminal s] -> Axon s
fromList = Unsorted . Seq.fromList


{- | Return all axon terminals, ordered by delay and target -}
terminals :: Axon s -> [Terminal s]
terminals axon@(Unsorted ss) = terminals $ sort axon
terminals axon@(Sorted _) = concat $ map wrap $ terminalsByDelay axon
    where
        wrap (d, ss) = map (\(tgt, w, s) -> AxonTerminal tgt d w s) ss



{- | Return all axon terminals.
 -
 - This function differs from 'terminals' in that the order of the synapses may
 - differ depending on how the axon has been modified previously. If the neuron
 - is currently unsorted the neurons will be returned in the same order in
 - which they were inserted. This method will be cheaper for unsorted axons -}
terminalsUnordered :: Axon s -> [Terminal s]
terminalsUnordered (Unsorted ss) = toList ss
terminalsUnordered axon@(Sorted _) = terminals axon


{- | Return all synapses, ordered by delay -}
terminalsByDelay :: Axon s -> [(Delay, [(Target, Weight, s)])]
terminalsByDelay axon =
    case axon of
        (Unsorted _)-> terminalsByDelay $ sort axon
        (Sorted ss) -> Assocs.mapElems synapsesD $ Map.toList ss


{- | Return number of synapses -}
size :: Axon s -> Int
size (Unsorted ss) = Seq.length ss
size (Sorted ss) = sum $ map sizeD $ Map.elems ss


maxDelay :: Axon s -> Delay
maxDelay (Unsorted ss) = L.maxOr0 $ toList $ fmap delay ss
maxDelay (Sorted ss) =
    if Map.null ss
        then 0
        else fst $ Map.findMax $ ss


{- | Return list of all targets, including duplicates -}
targets :: Axon s -> [Target]
targets (Unsorted ss) = toList $ fmap target ss
targets (Sorted ss) = concatMap targetsD $ Map.elems ss
    where
        targetsD ssD = concatMap targetsDT $ Map.assocs ssD
        targetsDT (tgt, ss) = replicate (length ss) tgt


{- | Add a synapse to axon. Duplicates are kept -}
connect :: Terminal s -> Axon s -> Axon s
{-# SPECIALIZE connect :: Terminal () -> Axon () -> Axon () #-}
connect s (Unsorted ss) = Unsorted $ ss |> s
connect s axon@(Sorted ss) = Sorted $ connectSorted s ss


connectSorted :: Terminal s -> AxonS s -> AxonS s
{-# SPECIALIZE connectSorted :: Terminal () -> AxonS () -> AxonS () #-}
connectSorted = connectSortedWith (++)


{- | Add a synapse with a specified combining function to use in case two
 - synapses have the same source, target, and delay -}
connectSortedWith :: ([Leaf s] -> [Leaf s] -> [Leaf s]) -> Terminal s -> AxonS s -> AxonS s
connectSortedWith f s ss = Map.alter (go (target s) (weight s) (atAux s)) (delay s) ss
    where
        go t w s Nothing = Just $ Map.singleton t [(w, s)]
        go t w s (Just ss) =
           let ss' = Map.insertWith f t [(w, s)] ss in ss' `seq` Just ss'


{- | Add a group of synapses -}
connectMany :: [Terminal s] -> Axon s -> Axon s
connectMany ss' (Unsorted ss) = Unsorted $ ss >< Seq.fromList ss'
connectMany ss' (Sorted ss) = Sorted $ foldl' (flip connectSorted) ss ss'


{- | Check if synapse is part of an axon -}
present :: (Eq s) => Terminal s -> AxonS s -> Bool
present s ss = isJust found
   where
       found = find eq =<< Map.lookup (target s) =<< Map.lookup (delay s) ss
       eq (w, pl) = w == weight s && pl == atAux s



{- | Remove the first matching synapse -}
disconnect :: (Eq s) => Terminal s -> Axon s -> Axon s
disconnect s axon =
    case axon of
        (Unsorted _) -> disconnect s $ sort axon
        (Sorted ss) ->
            if present s ss
                -- TODO: do lookup and delete in one go
                then Sorted $ Map.adjust (Map.adjust (delete ((weight s, atAux s))) (target s)) (delay s) ss
                else axon


{- | Remove the first matching synapse, reporting error in monad if no match is
 - found -}
disconnectM
    :: (Monad m, Eq s, Show s)
    => Terminal s -> Axon s -> m (Axon s)
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
    => Terminal s -> Terminal s -> Axon s -> m (Axon s)
replaceM old new axon =
    case axon of
        (Unsorted _) -> replaceM old new $ sort axon
        (Sorted ss) ->
            if present old ss
                then return $ connect new $ disconnect old axon
                else fail $ "Axon.replace: failed to find synapse " ++ show old



{- | Map function over all target indices -}
withTargets :: (Target -> Target) -> Axon s -> Axon s
withTargets f (Unsorted ss) = Unsorted $ fmap (withTarget f) ss
withTargets f (Sorted ss) = Sorted $ Map.map go ss
    where
        -- TODO: we should use same merging scheme as in 'connect'
        go  = Map.mapKeysWith err f
        err = error "Axon.withTargets: updated axon contains duplicate targets"


{- | Map function over all weights -}
withWeights :: (Weight -> Weight) -> Axon s -> Axon s
withWeights f (Unsorted ss) = Unsorted $ fmap (withWeight f) ss
withWeights f (Sorted ss) = Sorted $ Map.map (Map.map (map $ withLeafWeight f)) ss


-- TODO: make instance of Show instead
hPrintConnections :: (Show s) => Handle -> Source -> Axon s -> IO ()
hPrintConnections hdl src axon = do
    forM_ (terminalsByDelay axon) $ \(d, ss) -> do
        forM_ ss $ \(tgt, w, s) -> do
            hPutStrLn hdl $ (show src) ++ " -> " ++ (intercalate " " $ [show tgt, show d, show w, show s])


instance (Show s) => Show (Axon s) where
    {- Print one synapse per line -}
    showsPrec _ a s = showSynapses (terminals a) s
        where
            showSynapses [] = id
            showSynapses (s:ss) = shows s . showChar '\n' . showSynapses ss

instance (NFData s) => NFData (Terminal s) where
    rnf (AxonTerminal t d w a) = rnf t `seq` rnf d `seq` rnf w `seq` rnf a `seq` ()

instance (NFData s) => NFData (Axon s) where
    rnf (Unsorted ss) = (rnf $! toList ss) `seq` ()
    rnf (Sorted ss) = rnf ss `seq` ()


(|>) = (Seq.|>)
(><) = (Seq.><)
