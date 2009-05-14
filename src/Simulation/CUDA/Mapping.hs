{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}


module Simulation.CUDA.Mapping (
    mapNetwork,
    CuPartition,
        neurons,
        neuronAssocs,
    CuNet,
        partitionSizes,
        partitionCount,
        partitions,
        partitionAssocs,
        maxNetworkDelay,
        maxL0Width,
        maxL1Width
) where

import Control.Monad.Writer
import Data.Function (on)
import Data.List (insert, sortBy, groupBy, partition, foldl', sort)
import Data.Maybe (fromMaybe, isJust, fromJust)
import qualified Data.Map as Map

import Construction.Izhikevich (IzhNeuron, IzhState)
import qualified Construction.Network as Net
import qualified Construction.Neurons as Neurons (Neurons, empty)
import qualified Construction.Neuron as N
import Construction.Synapse
import Simulation.CUDA.Address
import Simulation.CUDA.KernelFFI as Kernel (maxPartitionSize)
import Types


{- During mapping we may want to log various network statistics -}
type Log = String

logMsg :: String -> Writer Log ()
logMsg msg = tell $ msg ++ "\n"



{- Create mapping in the form of an address translation table, where the
 - network has been split into a number of fixed-size groups. Return both the
 - mapping and the number of groups. -}
mapNetworkATT :: Int -> [Idx] -> (ATT, Int)
mapNetworkATT psize gidx = (mkATT ncount pcount psize gidx didx, pcount)
    where
        pidx = concatMap (replicate psize) [0..]
        nidx = concat $ repeat [0..psize-1]
        didx = zip pidx nidx
        ncount = length gidx
        pcount = divRound ncount psize
        divRound x y = x `div` y + (if x `mod` y == 0 then 0 else 1)


{- | The partition size the mapper should use depends on both the maximum
 - supported by the kernel (given some configuration) and the request of the
 - user. partitionSize determines this size and logs a message if the user
 - request is over-ridden.  -}
targetPartitionSize :: Bool -> Maybe Int -> Writer Log Int
targetPartitionSize usingSTDP userSz = do
    let maxSz = Kernel.maxPartitionSize usingSTDP
    maybe (return maxSz) (validateUserSz maxSz) userSz
    where
        validateUserSz maxSz userSz =
            if userSz <= maxSz
                then return userSz
                else do
                    logMsg $ "Invalid partition size (" ++ show userSz
                        ++ ") requested" ++ ", defaulting to " ++ show maxSz
                    return maxSz


type CuPartition n s = Map.Map NeuronIdx (N.Neuron n s)

{- | Return number of neurons in partition -}
partitionSize :: CuPartition n s -> Int
partitionSize = Map.size

neurons :: CuPartition n s -> [n]
neurons = map N.ndata . Map.elems

neuronAssocs :: CuPartition n s -> [(NeuronIdx, N.Neuron n s)]
neuronAssocs = Map.assocs


data CuNet n s = CuNet {
        netPartitions :: Map.Map PartitionIdx (CuPartition n s),
        maxNetworkDelay :: Int,
        maxL0Width :: Int,
        maxL1Width :: Int
    }

-- | Return list of sizes for each partition
partitionSizes :: CuNet n s -> [Int]
partitionSizes = map partitionSize . Map.elems . netPartitions


partitionCount :: CuNet n s -> Int
partitionCount = Map.size . netPartitions


partitions :: CuNet n s -> [CuPartition n s]
partitions = Map.elems . netPartitions


partitionAssocs :: CuNet n s -> [(PartitionIdx, CuPartition n s)]
partitionAssocs = Map.assocs . netPartitions


{- | Determine the max delay, as well as the L0 and L1 pitches for a particular
 - neuron -}
synapseParameters :: ATT -> PartitionIdx -> N.Neuron n s -> (Delay, Int, Int)
synapseParameters att preIdx n = (N.maxDelay n, l0, l1)
    where
        (l0, l1) = N.foldTarget go' (0, 0) n
        go' (pitchL0, pitchL1) s =
            if isL0 s
                then (pitchL0+1, pitchL1)
                else (pitchL0, pitchL1+1)

        isL0 t = (==preIdx) $! partitionIdx $! deviceIdx att t


networkInsert :: ATT -> CuNet n s -> (Idx, N.Neuron n s) -> CuNet n s
networkInsert att (CuNet ptn mxd l0w l1w) (gidx, n) =
    ptn' `seq` mxd' `seq` l0w' `seq` l1w' `seq` CuNet ptn' mxd' l0w' l1w'
    where
        ptn' = Map.alter clusterInsert pidx ptn

        didx = deviceIdx att gidx
        pidx = partitionIdx didx
        nidx = neuronIdx didx
        -- TODO: might want to gather all this information in the first pass
        (n_mxd, n_l0w, n_l1w) = synapseParameters att pidx n

        -- fold arguments into synapse parameters
        mxd' = max mxd n_mxd
        l0w' = max l0w n_l0w
        l1w' = max l1w n_l1w

        -- clusterInsert :: Maybe (CuPartition n) -> Maybe (CuPartition n)
        clusterInsert (Just c) = Just $! Map.insert nidx n c
        clusterInsert Nothing  = Just $! Map.singleton nidx n


cuNetwork :: ATT -> Neurons.Neurons n s -> CuNet n s
cuNetwork att ns = foldl' (networkInsert att) empty cns
    where
        cns = Map.toList ns
        empty = CuNet Neurons.empty 0 0 0


{- | Map network onto partitions containing the same number of neurons -}
mapNetwork
    :: Net.Network n s
    -> Bool          -- ^ are we using STDP?
    -> Maybe Int     -- ^ user-specified fixed size of each partition
    -- TODO: we may want to use Seq String here instead
    -> Writer Log (CuNet n s, ATT)
mapNetwork net@(Net.Network ns _) usingSTDP userPSz = do
    n <- targetPartitionSize usingSTDP userPSz
    logMsg $ "Using partition size of " ++ show n
    let (att, nclusters) = mapNetworkATT n $ Net.indices net
        d_ns = cuNetwork att ns
    return (d_ns, att)
