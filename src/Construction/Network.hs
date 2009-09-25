{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Construction.Network (
        Network(..),
        empty,
        -- * Query
        size,
        isEmpty,
        synapseCount,
        indices,
        idxBounds,
        synapses,
        weightMatrix,
        synapsesOf,
        neurons,
        toList,
        maxDelay,
        maxSynapsesPerNeuron,
        -- * Modify
        addNeuron,
        addNeuronGroup,
        withNeurons,
        withTerminals,
        -- * Pretty-printing
        printConnections,
        hPrintConnections,
        printNeurons,
        hPrintNeurons
    ) where

import Control.Parallel.Strategies (NFData, rnf)
import qualified Data.Map as Map
import System.IO (Handle, stdout)

import qualified Construction.Neuron as Neuron
import qualified Construction.Neurons as Neurons
import Construction.Synapse
import Construction.Topology
import Types



{- For the synapses we just store the indices of pre and post. The list should
 - be sorted to simplify the construction of the in-memory data later. -}
data Network n s = Network {
        networkNeurons :: !(Neurons.Neurons n s),
        topology :: !(Topology Idx)
    } deriving (Eq, Show)


empty = Network (Neurons.empty) NoTopology


-------------------------------------------------------------------------------
-- Query
-------------------------------------------------------------------------------

{- | Return number of neurons in the network -}
size :: Network n s -> Int
size = Neurons.size . networkNeurons


isEmpty :: Network n s -> Bool
isEmpty = Neurons.isEmpty . networkNeurons


{- | Return total number of synapses in the network -}
synapseCount :: Network n s -> Int
synapseCount = Neurons.synapseCount . networkNeurons


{- | Return indices of all valid neurons -}
indices :: Network n s -> [Idx]
indices = Neurons.indices . networkNeurons


{- | Return minimum and maximum neuron indices -}
idxBounds :: Network n s -> (Idx, Idx)
idxBounds = Neurons.idxBounds . networkNeurons


{- | Return synapses orderd by source and delay -}
synapses :: Network n s -> [(Idx, [(Delay, [(Idx, Current, s)])])]
synapses = Neurons.synapses . networkNeurons


synapsesOf :: Network n s -> Idx -> [Synapse s]
synapsesOf = Neurons.synapsesOf . networkNeurons


{- | Return synapses organised by source only -}
weightMatrix :: Network n s -> Map.Map Idx [Synapse s]
weightMatrix = Neurons.weightMatrix . networkNeurons


{- | Return list of all neurons -}
neurons :: Network n s -> [Neuron.Neuron n s]
-- TODO: merge Neurons into Network, this is just messy!
neurons = Neurons.neurons . networkNeurons


toList = Neurons.toList . networkNeurons


{- | Return maximum delay in network -}
maxDelay :: Network n s -> Delay
maxDelay = Neurons.maxDelay . networkNeurons


maxSynapsesPerNeuron :: Network n s -> Int
maxSynapsesPerNeuron = Neurons.maxSynapsesPerNeuron . networkNeurons



-------------------------------------------------------------------------------
-- Modification
-------------------------------------------------------------------------------


addNeuron idx n = withNeurons (Neurons.addNeuron idx n)

{- | Add group of neurons, lazily -}
addNeuronGroup ns = withNeurons (Neurons.addNeuronGroup ns)


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



{- | Apply function to all weights -}
withWeights :: (Current -> Current) -> Network n s -> Network n s
withWeights f (Network ns t) = Network ns' t
    where
        ns' = Neurons.withWeights f ns


-------------------------------------------------------------------------------
-- Various
-------------------------------------------------------------------------------


instance (NFData n, NFData s) => NFData (Network n s) where
    rnf (Network n t) = rnf n `seq` rnf t


-------------------------------------------------------------------------------
-- Printing
-------------------------------------------------------------------------------

hPrintConnections :: (Show s) => Handle -> Network n s -> IO ()
hPrintConnections hdl = Neurons.hPrintConnections hdl . networkNeurons

printConnections :: (Show s) => Network n s -> IO ()
printConnections = hPrintConnections stdout


printNeurons :: (Show n, Show s) => Network n s -> IO ()
printNeurons = hPrintNeurons stdout

hPrintNeurons :: (Show n, Show s) => Handle -> Network n s -> IO ()
hPrintNeurons hdl = Neurons.hPrintNeurons hdl . networkNeurons

