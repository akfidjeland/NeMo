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
        fromList,
        unconnected,
        -- * Query
        synapses,
        synapsesByDelay,
        size,
        maxDelay,
        -- * Modify
        connect,
        connectMany,
        disconnect,
        disconnectM,
        replaceM,
        withTargets,
        -- * Traverse
        foldTarget,
        -- * Pretty-print
        printConnections,
        -- * Internal
        strip
    ) where


import Control.Monad (forM_, liftM)
import Data.Binary
import Data.List (foldl', delete, find, intercalate)
import qualified Data.Map as Map
import Data.Maybe (isJust)
import Control.Parallel.Strategies (NFData, rnf, using)

-- import Construction.Synapse (Synapse(..), delay)
import Construction.Synapse (Synapse(..), delay, Static)
import Types (Idx, Delay)
import qualified Util.List as L (replace)


{- Synapses are stored sorted by delay, since this is how the backend will need
 - to access them. We only need the postsynaptic index and the "payload" (which
 - varies depending on synapse type. The delay and the presynaptic is stored in
 - the collection data structure. -}

newtype Axon s = Axon (Map.Map Delay [SData s]) deriving (Eq, Show)

smap (Axon ss) = ss

{- Some neuron data is stored in the data structure, rather than in the nodes,
 - so we use a reduced representation. -}
type SData s = (Idx, s)


{- | Extract only the synapse data used stored in the leaf nodes -}
strip :: Synapse s -> SData s
strip s = tgt `seq` pl `seq` (tgt, pl)
    where
        tgt = target s
        pl  = sdata s


{- | Create axon from a list of synapses -}
-- fromList :: (Synaptic s ix r) => [Synapse s] -> Axon s
fromList :: [Synapse s] -> Axon s
fromList = foldl' (flip connect) unconnected


{- | Return axon with no synapses -}
unconnected :: Axon s
unconnected = Axon Map.empty


{- | Return all synapses, not ordered -}
synapses :: Idx -> Axon s -> [Synapse s]
synapses src axon = concat $ map wrap $ synapsesByDelay axon
    where
        wrap (d, ss) = map (\(tgt, s) -> Synapse src tgt d s) ss


{- | Return all synapses, ordered by delay -}
synapsesByDelay :: Axon s -> [(Delay, [(Idx, s)])]
synapsesByDelay (Axon ss) = Map.toList ss


{- | Return number of synapses -}
size :: Axon s -> Int
size = length . concat . Map.elems . smap


maxDelay :: Axon s -> Delay
maxDelay = fst . Map.findMax . smap


{- | Add a synapse -}
connect :: Synapse s -> Axon s -> Axon s
connect s (Axon ss) = s' `seq` Axon $ Map.alter (go s') (delay s) ss
    where
        s' = strip s
        go s'' Nothing   = Just [s'']
        go s'' (Just ss) = let ss' = (s'':ss) in ss' `seq` Just ss'


{- | Add a group of synapses -}
connectMany :: [Synapse s] -> Axon s -> Axon s
connectMany ss axon = foldl' (flip connect) axon ss


{- | Check if synapse is part of an axon -}
present :: (Eq s) => Synapse s -> Axon s -> Bool
present s (Axon ss) = isJust (find (==(strip s)) =<< Map.lookup (delay s) ss)


{- | Remove the first matching synapse -}
disconnect :: (Eq s) => Synapse s -> Axon s -> Axon s
disconnect s (Axon ss) = Axon $ Map.adjust (delete (strip s)) (delay s) ss


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
withTargets :: (Idx -> Idx) -> Axon s -> Axon s
withTargets f (Axon ss) = Axon $ Map.map (map (\(t, s) -> (f t, s))) ss


{- | Fold function over all target indices -}
foldTarget :: (a -> Idx -> a) -> a -> Axon s -> a
foldTarget f x (Axon ss) = Map.fold (flip (foldl' f')) x ss
    where
        f' x s = f x (fst s)


-- TODO: make instance of Show instead
printConnections :: (Show s) => Idx -> Axon s -> IO ()
printConnections src axon = do
    forM_ (synapsesByDelay axon) $ \(d, ss) -> do
        forM_ ss $ \(tgt, s) -> do
            putStrLn $ intercalate " " $ [show src, show tgt, show d, show s]


instance (Binary s) => Binary (Axon s) where
    put (Axon ss) = put (Map.size ss) >> mapM_ put (Map.toAscList ss)
    -- get = liftM (Axon . Map.fromDistinctAscList) get
    get = do
        ss <- liftM Map.fromAscList get
        -- TODO: could force evaluation of list elements instead
        -- make sure to walk over the whole map to force evaluation
        return $! Axon $! snd $! Map.mapAccum f' () ss
        where
            f' () x = x `seq` ((), x)

instance (NFData s) => NFData (Axon s) where
    rnf (Axon ss) = rnf ss
