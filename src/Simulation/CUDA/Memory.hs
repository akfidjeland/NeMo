{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}

module Simulation.CUDA.Memory (
    initMemory,
    SimData(..),
    -- * Opaque pointer to kernel data structures
    CuRT
) where


import Control.Monad (zipWithM_, forM_)
import Data.Array.MArray (newListArray)
import Data.Maybe (Maybe, isNothing)
import Foreign.C.Types (CSize, CFloat, CUInt)
import Foreign.ForeignPtr
import Foreign.Marshal.Array (mallocArray)
import Foreign.Ptr
import Foreign.Storable (pokeElemOff)

import Construction.Neuron (synapsesByDelay)
import Construction.Izhikevich
import Construction.Synapse (Static, target, current)
import Simulation.CUDA.Address
import Simulation.CUDA.KernelFFI
import Simulation.CUDA.Mapping
import Simulation.STDP
import Types


data SimData = SimData {
        ccount  :: Int,
        csize   :: [Int],              -- size of each cluster,
        att     :: ATT,
        rt      :: ForeignPtr CuRT          -- kernel runtime data
    }



{- Initialise memory on a single device -}
initMemory
    :: CuNet (IzhNeuron FT) Static
    -> ATT
    -> Int
    -> Maybe STDPConf
    -> IO SimData
initMemory net att maxProbePeriod stdp = do
    (pcount, psizes, rt) <- allocRT net maxProbePeriod
    configureSTDP rt stdp
    loadAllNeurons rt net
    loadCMatrix rt att net
    return $ SimData pcount psizes att rt


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


-------------------------------------------------------------------------------
-- Runtime data
-------------------------------------------------------------------------------

foreign import ccall unsafe "allocRuntimeData"
    c_allocRT
        :: CSize  -- ^ partition count
        -> CSize  -- ^ max partition size
        -> CUInt  -- ^ max delay (L0 and L1)
        -> CSize  -- ^ max L0 synapses per delay
        -> CSize  -- ^ max L0 synapses per delay in reverse matrix
        -> CSize  -- ^ max L1 synapses per delay
        -> CSize  -- ^ max L1 synapses per delay in reverse matrix
        -> CSize  -- ^ l1 spike queue entry size
        -> CUInt  -- ^ max read period
        -> IO (Ptr CuRT)

foreign import ccall unsafe "&freeRuntimeData"
    c_freeRT :: FunPtr (Ptr CuRT -> IO ())

allocRT :: CuNet n s -> Int -> IO (Int, [Int], ForeignPtr CuRT)
allocRT net maxProbePeriod = do
    let pcount = partitionCount net
        psizes = partitionSizes net
    ptr <- c_allocRT
        (fromIntegral pcount)
        (fromIntegral $! maximum psizes)
        (fromIntegral $! maxNetworkDelay net)
        (fromIntegral $! maxL0Pitch net)
        (fromIntegral $! maxL0RPitch net)
        (fromIntegral $! maxL1Pitch net)
        (fromIntegral $! maxL1RPitch net)
        -- TODO: compute properly how large the buffers should be
        64768 -- L1 queue size
        (fromIntegral maxProbePeriod)
    rt <- newForeignPtr c_freeRT ptr
    return (pcount, psizes, rt)


configureSTDP :: ForeignPtr CuRT -> Maybe STDPConf -> IO ()
configureSTDP _ Nothing = return ()
configureSTDP rt (Just conf) =
    enableSTDP rt
        (stdpTauP conf) (stdpTauD conf)
        (stdpAlphaP conf) (stdpAlphaD conf)
        (stdpMaxWeight conf)
