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
        maxL0Pitch,
        maxL0RPitch,
        maxL1Pitch,
        maxL1RPitch
) where

import Control.Monad.Writer
import Control.Monad.ST
import Data.Array.ST
import Data.Function (on)
import Data.List (insert, sortBy, groupBy, partition, foldl', sort)
import Data.Maybe (fromMaybe, isJust, fromJust)
import qualified Data.Map as Map

import Construction.Izhikevich (IzhNeuron, IzhState)
import qualified Construction.Network as Net
import qualified Construction.Neurons as Neurons (Neurons, empty, size)
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
        netPartitions   :: Map.Map PartitionIdx (CuPartition n s),
        maxNetworkDelay :: Int,
        -- | Pitch of connectivity matrix, i.e. max synapses for any single
        -- presynaptic neuron and delay
        maxL0Pitch      :: Int,
        maxL1Pitch      :: Int,
        maxL0RPitch     :: Int,
        maxL1RPitch     :: Int
        -- TODO: also determine max L1 spike buffer size
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



type RPitch t = STUArray t (PartitionIdx, NeuronIdx, Delay) Int


mkRPitch stdp pcount psize =
    if stdp
        then do
            l0r <- arr
            l1r <- arr
            return $! Just (l0r, l1r)
        else return Nothing
    where
        -- TODO: remove hard-coding of max delay hee
        arr = newArray ((0,0,1), (pcount-1, psize-1, 32)) 0



{- Increment either L0 or L1 reverse -}
accRPitch :: ATT -> DeviceIdx -> (RPitch t, RPitch t) -> Synapse s -> ST t ()
accRPitch att src@(srcp,_) (l0r, l1r) s =
    if isL0
        then inc l0r src d
        else inc l1r src d
    where
        d   = delay s
        isL0 = (==srcp) $! partitionIdx $! deviceIdx att $! target s
        inc ss (p,n) d = readArray ss i >>= writeArray ss i . (+1)
            where i = (p,n,d)


maxRPitch :: RPitch t -> ST t Int
maxRPitch arr = return . maximum =<< getElems arr


networkInsert
    :: ATT
    -> Maybe (RPitch t, RPitch t)
    -> CuNet n s
    -> (Idx, N.Neuron n s)
    -> ST t (CuNet n s)
networkInsert att rpitch (CuNet ptn mxd l0w l1w l0r l1r) (gidx, n) = do
    -- TODO: use synapsesByDelay here instead.
    maybe (return ()) (\r -> mapM_ (accRPitch att didx r) $ N.synapses gidx n) rpitch
    return $! ptn' `seq` mxd' `seq` l0w' `seq` l1w' `seq` CuNet ptn' mxd' l0w' l1w' l0r l1r
    where
        ptn' = Map.alter clusterInsert pidx ptn

        didx = deviceIdx att gidx
        pidx = partitionIdx didx
        nidx = neuronIdx didx

        {- Note: We cannot determine these parameters on the forward pass, as
         - we won't know which synapses are L0 and which are L1 before we do
         - the mapping. We'd have to interleave the mapping with the
         - construction. It would, however, be possible to determine the
         - global maximum. It would then be possible to set per-neuron maxima
         - for L0 and L1 when we load this onto the device. -}
        (n_mxd, n_l0w, n_l1w) = synapseParameters att pidx n

        -- fold arguments into synapse parameters
        mxd' = max mxd n_mxd
        l0w' = max l0w n_l0w
        l1w' = max l1w n_l1w

        -- clusterInsert :: Maybe (CuPartition n) -> Maybe (CuPartition n)
        clusterInsert (Just c) = Just $! Map.insert nidx n c
        clusterInsert Nothing  = Just $! Map.singleton nidx n


cuNetwork :: ATT -> Bool -> Int -> Int -> Neurons.Neurons n s -> CuNet n s
cuNetwork att stdp pcount psize ns = runST $ do
    rpitch <- mkRPitch stdp pcount psize
    net <- foldM (networkInsert att rpitch) empty cns
    l0r <- maybe (return 0) (maxRPitch . fst) rpitch
    l1r <- maybe (return 0) (maxRPitch . snd) rpitch
    return $ net {
            maxL0RPitch = l0r,
            maxL1RPitch = l1r
        }
    where
        cns = Map.toList ns
        empty = CuNet Map.empty 0 0 0 0 0


{- | Map network onto partitions containing the same number of neurons -}
mapNetwork
    :: Net.Network n s
    -> Bool          -- ^ are we using STDP?
    -> Maybe Int     -- ^ user-specified fixed size of each partition
    -- TODO: we may want to use Seq String here instead
    -> Writer Log (CuNet n s, ATT)
mapNetwork net@(Net.Network ns _) stdp psizeReq = do
    psize <- targetPartitionSize stdp psizeReq
    let (att, pcount) = mapNetworkATT psize $ Net.indices net
        d_ns = cuNetwork att stdp pcount psize ns
    logMsg $ "Network size: " ++ show (Net.size net)
    logMsg $ "Partition count: " ++ show pcount
    logMsg $ "Partition size: " ++ show psize
    logMsg $ "L0 pitch: " ++ show (maxL0Pitch d_ns)
    logMsg $ "L1 pitch: " ++ show (maxL1Pitch d_ns)
    when stdp $ logMsg $ "L0 reverse pitch: " ++ show (maxL0RPitch d_ns)
    when stdp $ logMsg $ "L1 reverse pitch: " ++ show (maxL1RPitch d_ns)
    return (d_ns, att)
