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
import Data.Maybe

import Types


type PartitionIdx = Int
type NeuronIdx = Int
type DeviceIdx = (PartitionIdx, NeuronIdx)

partitionIdx :: DeviceIdx -> PartitionIdx
partitionIdx = fst

neuronIdx :: DeviceIdx -> NeuronIdx
neuronIdx = snd



data ATT = ATT {
        g2d :: Array Idx (Maybe DeviceIdx),
        d2g :: Array DeviceIdx (Maybe Idx)
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
        gm = accumArray merge Nothing (0, ncount-1) $ zip gs $ map Just ds
        dm = accumArray merge Nothing ((0,0), (pcount-1, psize-1)) $ zip ds $ map Just gs

        merge Nothing (Just a) = Just a
        merge e1 e2 = error $ "Duplicate neuron indices in ATT construction: " ++ show e1 ++ " and " ++ show e2


-- Unsafe lookup
globalIdx :: ATT -> DeviceIdx -> Idx
globalIdx att idx = fromJust $! (d2g att) ! idx

globalIdxM :: (Monad m) => ATT -> DeviceIdx -> m Idx
globalIdxM att = idxLookup "device" $ d2g att


-- Unsafe lookup
deviceIdx :: ATT -> Idx -> DeviceIdx
deviceIdx att idx = fromJust $! (g2d att) ! idx

deviceIdxM :: (Monad m) => ATT -> Idx -> m DeviceIdx
deviceIdxM att = idxLookup "global" $ g2d att


-- | Safe array lookup
idxLookup :: (Monad m, Ix a, Show a) => String -> Array a (Maybe b) -> a -> m b
idxLookup from arr idx =
    if inRange (bounds arr) idx
        then maybe (fail $ "invalid neuron: " ++ show idx) return $ arr!idx
        else fail $ "invalid " ++ from ++ " index (" ++ show idx ++ ")"
