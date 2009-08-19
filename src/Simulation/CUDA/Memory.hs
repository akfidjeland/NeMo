{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}

module Simulation.CUDA.Memory (
    initMemory,
    getWeights
) where


import Control.Monad (zipWithM_, zipWithM, forM_)
import Data.Array.MArray (newListArray)
import Data.Maybe (Maybe, isNothing)
import qualified Data.Map as Map (Map, fromList)
import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Marshal.Array (mallocArray)
import Foreign.Ptr
import Foreign.Storable (pokeElemOff, peekElemOff)

import Construction.Neuron (synapsesByDelay)
import Construction.Izhikevich (IzhNeuron(..), stateSigma)
import Construction.Synapse (Synapse(..), Static(..), target, current)
import Simulation.CUDA.Address
import Simulation.CUDA.KernelFFI
import Simulation.CUDA.State (State(..))
import Simulation.CUDA.Mapping
import Simulation.STDP
import Types





{- Initialise memory on a single device -}
initMemory
    :: CuNet (IzhNeuron FT) Static
    -> ATT
    -> Int
    -> Int
    -> StdpConf
    -> IO State
initMemory net att maxProbePeriod dt stdp = do
    (pcount, psizes, maxDelay, rt) <- allocRT net maxProbePeriod
    configureStdp rt stdp
    loadAllNeurons rt net
    loadCMatrix rt att net
    return $ State pcount psizes maxDelay (fromIntegral dt) att rt


loadPartitionNeurons
    :: Ptr CuRT
    -> CSize
    -> CuPartition (IzhNeuron FT) Static
    -> IO ()
loadPartitionNeurons rt pidx partition = do
    forPartition loadA $ listOf paramA
    forPartition loadB $ listOf paramB
    forPartition loadC $ listOf paramC
    forPartition loadD $ listOf paramD
    forPartition loadU $ listOf stateU
    forPartition loadV $ listOf stateV
    let sigma = allJust 0 $ listOf stateSigma
    maybe (return ()) (forPartition loadThalamicInputSigma) sigma
    where
        listOf f = map f ns
        forPartition load xs = do
            arr <- newListArray (0, len-1) $ map realToFrac xs
            load rt pidx len arr
        ns = neurons partition
        -- TODO: just read the partition size
        len = length ns


loadAllNeurons :: ForeignPtr CuRT -> CuNet (IzhNeuron FT) Static -> IO ()
loadAllNeurons rt net =
    withForeignPtr rt $ \rtptr -> do
    zipWithM_ (loadPartitionNeurons rtptr) [0..] $ partitions net


{- | Return just a list with Nothing replaced by default value. If all are
 - nothing, return Nothing -}
allJust :: a -> [Maybe a] -> Maybe [a]
allJust d xs
    | all isNothing xs = Nothing
    | otherwise = Just $ map toDefault xs
    where
        toDefault Nothing = d
        toDefault (Just x) = x


-------------------------------------------------------------------------------
-- Connectivity matrix
-------------------------------------------------------------------------------


{- | By the time we write data to the device, we have already established the
 - maximum pitch for the connectivity matrices. The data is written on a
 - per-row basis. To avoid excessive memory allocation we allocate a single
 - buffer (for each CM) with the know maximum pitch, and re-use it for each
 - row. -}
data Outbuf = Outbuf {
        weights :: Ptr CFloat,
        pidx    :: Ptr CUInt,
        nidx    :: Ptr CUInt
    }


allocOutbuf len = do
    wbuf <- mallocArray len
    pbuf <- mallocArray len
    nbuf <- mallocArray len
    return $! Outbuf wbuf pbuf nbuf


{- | Write all connectivity data to device -}
loadCMatrix rt att net =
    withForeignPtr rt $ \rtptr -> do
    bufL0 <- allocOutbuf $ maxL0Pitch net
    bufL1 <- allocOutbuf $ maxL1Pitch net
    forM_ (partitionAssocs net) $ \(pidx, p) -> do
        forM_ (neuronAssocs p) $ \(nidx, n) -> do
            forM_ (synapsesByDelay n) $ \(delay, ss) -> do
                let -- isL0 :: (Idx, s) -> Bool
                    -- TODO: force evaluation?
                    isL0 = ((==) pidx) . partitionIdx . deviceIdx att . fst
                (len0, len1) <- pokeSynapses bufL0 0 bufL1 0 att isL0 ss
                setRow rtptr bufL0 cmatrixL0 (pidx, nidx) delay len0
                setRow rtptr bufL1 cmatrixL1 (pidx, nidx) delay len1
    where
        setRow rtptr buf = setCMDRow rtptr (weights buf) (pidx buf) (nidx buf)




{- | Write a row of synapses (i.e for a single presynaptic/delay) to output buffer -}
pokeSynapses _ len0 _ len1 _ _ [] = return (len0, len1)
pokeSynapses buf0 i0 buf1 i1  att isL0 (s:ss) = do
    if isL0 s
        then do
            pokeSynapse buf0 i0 att s
            pokeSynapses buf0 (i0+1) buf1 i1 att isL0 ss
        else do
            pokeSynapse buf1 i1 att s
            pokeSynapses buf0 i0 buf1 (i1+1) att isL0 ss


{- | Write a single synapse to output buffer -}
pokeSynapse :: Outbuf -> Int -> ATT -> (Idx, Static) -> IO ()
pokeSynapse buf i att (target, s) = do
    pokeElemOff (weights buf) i $! realToFrac $! current s
    let didx = deviceIdx att target
    pokeElemOff (pidx buf) i $! fromIntegral $! partitionIdx didx
    pokeElemOff (nidx buf) i $! fromIntegral $! neuronIdx didx


type DArr = (Ptr CInt, Ptr CInt, Ptr CFloat, Int)

pitch :: DArr -> Int
pitch (_, _, _, p) = p


{- | Get (possibly modified) connectivity matrix back from device -}
getWeights :: State -> IO (Map.Map Idx [Synapse Static])
getWeights sim = do
    withForeignPtr (rt sim) $ \rt_ptr -> do
    darr0 <- getCM rt_ptr cmatrixL0
    darr1 <- getCM rt_ptr cmatrixL1
    ns <- peekPartitions
            (globalIdx $ att sim)
            (pcount sim)
            (psize sim)
            (maxDelay sim)
            darr0 darr1
    return $! Map.fromList $ concat ns


-- for each partition in network
peekPartitions globalIdx pcount psizes d_max darr0 darr1 = zipWithM go [0..] psizes
    where
        go p_idx psize = do
            let s_idx0 = poffset p_idx psize darr0
            let s_idx1 = poffset p_idx psize darr1
            peekNeurons globalIdx p_idx 0 psize d_max s_idx0 s_idx1 darr0 darr1
        poffset p psize darr = p * maxPSize * d_max * pitch darr
        maxPSize = maximum psizes


-- for each neuron in partition
peekNeurons globalIdx p_idx n_idx n_max d_max s_idx0 s_idx1 darr0 darr1 =
    go 0 s_idx0 s_idx1
    where
        go n_idx s_idx0 s_idx1
            -- TODO have some way to determine end of partition, for small
            -- partitions
            | n_idx == n_max = return []
            | otherwise      = do
                let n_gidx = globalIdx (p_idx, n_idx)
                ss <- peekDelays globalIdx n_gidx d_max s_idx0 s_idx1 darr0 darr1
                ns <- go (n_idx+1) (step s_idx0 darr0) (step s_idx1 darr1)
                return $! (n_gidx, concat ss) : ns
        step s_idx darr = s_idx + d_max * pitch darr


-- for each delay in neuron
peekDelays globalIdx n_idx d_max s_idx0 s_idx1 darr0 darr1 = go 1 s_idx0 s_idx1
    where
        go d s_idx0 s_idx1
            | d > d_max = return []
            | otherwise  = do
                s0  <- peekAxon globalIdx n_idx d s_idx0 darr0
                s1  <- peekAxon globalIdx n_idx d s_idx1 darr1
                ss  <- go (d+1) (s_idx0 + pitch darr0) (s_idx1 + pitch darr1)
                return $! s0 : s1 : ss


{- | Get synapses for a specific delay -}
peekAxon
    :: (DeviceIdx -> Idx)
    -> Source
    -> Delay
    -> Int        -- ^ current synapse
    -> DArr       -- ^ device data
    -> IO [Synapse Static]
peekAxon globalIdx source d i darr = go i (i + pitch darr)
    where
        go i end
            | i == end  = return []
            | otherwise = do
                s  <- peekSynapse globalIdx i source d darr
                ss <- go (i+1) end
                case s of
                    -- TODO: ok to assume all null synapses at end
                    Nothing -> return $! ss
                    Just s  -> return $! (s:ss)


{- | Synapses pointing to the null neuron are considered inactive -}
-- TODO: get this value from kernel
nullIdx = (== (-1))


{- | Get a single synapse out of c-array -}
peekSynapse
    :: (DeviceIdx -> Idx)
    -> Int
    -> Source
    -> Delay
    -> DArr
    -> IO (Maybe (Synapse Static))
peekSynapse globalIdx i source delay (tp_arr, tn_arr, w_arr, _) = do
    tp <- peekElemOff tp_arr i
    if nullIdx tp
        then return $! Nothing
        else do
            tn     <- peekElemOff tn_arr i
            weight <- peekElemOff w_arr i
            let target = globalIdx (fromIntegral tp, fromIntegral tn)
            return $! Just $!
                Synapse source target delay $! Static (realToFrac weight)

-------------------------------------------------------------------------------
-- Runtime data
-------------------------------------------------------------------------------

foreign import ccall unsafe "allocRuntimeData"
    c_allocRT
        :: CSize  -- ^ partition count
        -> CSize  -- ^ max partition size
        -> CUInt  -- ^ max delay (L0 and L1)
        -> CSize  -- ^ max L0 synapses per delay
        -> CSize  -- ^ max L0 synapses per neuron in reverse matrix
        -> CSize  -- ^ max L1 synapses per delay
        -> CSize  -- ^ max L1 synapses per neuron in reverse matrix
        -> CSize  -- ^ l1 spike queue entry size
        -> CUInt  -- ^ max read period
        -> IO (Ptr CuRT)

foreign import ccall unsafe "&freeRuntimeData"
    c_freeRT :: FunPtr (Ptr CuRT -> IO ())

allocRT :: CuNet n s -> Int -> IO (Int, [Int], Delay, ForeignPtr CuRT)
allocRT net maxProbePeriod = do
    let pcount = partitionCount net
        psizes = partitionSizes net
        dmax   = maxNetworkDelay net
    ptr <- c_allocRT
        (fromIntegral pcount)
        (fromIntegral $! maximum psizes)
        (fromIntegral $! dmax)
        (fromIntegral $! maxL0Pitch net)
        (fromIntegral $! maxL0RPitch net)
        (fromIntegral $! maxL1Pitch net)
        (fromIntegral $! maxL1RPitch net)
        -- TODO: compute properly how large the buffers should be
        64768 -- L1 queue size
        (fromIntegral maxProbePeriod)
    rt <- newForeignPtr c_freeRT ptr
    return (pcount, psizes, dmax, rt)
