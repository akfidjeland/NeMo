{- | Addressing of neurons on the device
 -
 - When the network is constructed, each neuron has a unique address. The user
 - specifies probes etc. in terms of these addresses. The network may be
 - transformed however, when mapping onto the simulation backend. We therefore
 - need a map from the original indices to the backend-specific indices.
 -
 - On the device neurons are grouped in clusters. The addressing of neurons is
 - therefore hierarchical. The size of each cluster is architecture-dependent.
 - We use a tuple here rather than a custom data type, in order to auto-derive
 - Ix (for mapping onto flat address space), saving us some effort. -}

module Simulation.CUDA.Address (
    -- * Hierarchical addressing
    PartitionIdx,
    NeuronIdx,
    DeviceIdx,
    neuronIdx,
    partitionIdx,
    -- * Address translation
    ATT,
    mkATT,
    globalIdx,
    globalIdxM,
    deviceIdx,
    deviceIdxM
) where

import Control.Monad
import Data.Array

import Types


type PartitionIdx = Int
type NeuronIdx = Int
type DeviceIdx = (PartitionIdx, NeuronIdx)

partitionIdx :: DeviceIdx -> PartitionIdx
partitionIdx = fst

neuronIdx :: DeviceIdx -> NeuronIdx
neuronIdx = snd


data ATT = ATT {
        g2d :: Array Idx DeviceIdx,
        d2g :: Array DeviceIdx Idx
    }


mkATT
    :: Int          -- ^ Max neurons
    -> Int          -- ^ Max partition size
    -> Int          -- ^ Max partitions
    -> [Idx]
    -> [DeviceIdx]
    -> ATT
mkATT ncount pcount psize gs ds = ATT gm dm
    where
        gm = array (0, ncount-1)$ zip gs ds
        dm = array ((0,0), (pcount-1, psize-1)) $ zip ds gs


globalIdx :: ATT -> DeviceIdx -> Idx
globalIdx att idx = (d2g att) ! idx

globalIdxM :: (Monad m) => ATT -> DeviceIdx -> m Idx
globalIdxM att = idxLookup "globalIdx" $ d2g att


deviceIdx :: ATT -> Idx -> DeviceIdx
deviceIdx att idx = (g2d att) ! idx

deviceIdxM :: (Monad m) => ATT -> Idx -> m DeviceIdx
deviceIdxM att = idxLookup "deviceIdx" $ g2d att


-- | Safe array lookup
idxLookup name arr idx =
    if inRange (bounds arr) idx
        then return $ arr ! idx
        else error $ name ++ ": index not found " ++ show idx
