{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Construction.Network (
        Network(..),
        -- * Query
        size,
        indices,
        idxBounds,
        synapses,
        maxDelay,
        -- * Modify
        withNeurons,
        withTerminals,
        -- * Pretty-printing
        printConnections,
        printNeurons
    ) where

import Control.Monad (liftM2)
import Control.Parallel.Strategies (NFData, rnf)
import Data.Binary
import qualified Data.Map as Map
import Data.Maybe (fromJust)

import qualified Construction.Neuron as Neuron
import qualified Construction.Neurons as Neurons
import Construction.Synapse
import Construction.Topology
import Types



{- For the synapses we just store the indices of pre and post. The list should
 - be sorted to simplify the construction of the in-memory data later. -}
data Network n s = Network {
        neurons     :: Neurons.Neurons n s,
        topology    :: Topology Idx
    } deriving (Eq, Show)


-------------------------------------------------------------------------------
-- Query
-------------------------------------------------------------------------------

{- | Return number of neurons in the network -}
size :: Network n s -> Int
size = Neurons.size . neurons


{- | Return indices of all valid neurons -}
indices :: Network n s -> [Idx]
indices = Neurons.indices . neurons


{- | Return minimum and maximum neuron indices -}
idxBounds :: Network n s -> (Idx, Idx)
idxBounds = Neurons.idxBounds . neurons


{- | Return synapses orderd by source and delay -}
synapses :: Network n s -> [(Idx, [(Delay, [(Idx, s)])])]
synapses = Neurons.synapses . neurons


{- | Return maximum delay in network -}
maxDelay :: Network n s -> Delay
maxDelay = Neurons.maxDelay . neurons



-------------------------------------------------------------------------------
-- Modification
-------------------------------------------------------------------------------


{- | Apply function to all neurons -}
-- TODO: perhaps use Neuron -> Neuron instead
withNeurons :: (Neurons.Neurons n s -> Neurons.Neurons n s) -> Network n s -> Network n s
withNeurons f (Network ns t) = (Network (f ns) t)


{- | Map function over all terminals (source and target) of all synapses -}
withTerminals :: (Idx -> Idx) -> Network n s -> Network n s
withTerminals f (Network ns t) = Network ns' t'
    where
        ns' = Neurons.withTerminals f ns
        t'  = fmap f t



-------------------------------------------------------------------------------
-- Various
-------------------------------------------------------------------------------

instance (Binary n, Binary s) => Binary (Network n s) where
    put (Network ns t) = put ns >> put t
    get = liftM2 Network get get


instance (NFData n, NFData s) => NFData (Network n s) where
    rnf (Network n t) = rnf n `seq` rnf t


printConnections :: (Show s) => Network n s -> IO ()
printConnections net =
    mapM_ (uncurry Neuron.printConnections) $ Map.assocs $ neurons net


printNeurons :: (Show n, Show s) => Network n s -> IO ()
printNeurons net = mapM_ f $ Map.assocs $ neurons net
    where f (idx, n) = putStrLn $ show idx ++ " " ++ show n
