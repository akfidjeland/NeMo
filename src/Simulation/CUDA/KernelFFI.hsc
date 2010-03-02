{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Simulation.CUDA.KernelFFI (
    allocateRuntime,
    stepBuffering,
    stepNonBuffering,
    readFiring,
    applyStdp,
    -- TODO: rename
    getCMDRow,
    startSimulation,
    syncSimulation,
    printCycleCounters,
    freeRT,
    CuRT,
    addNeuron,
    addSynapses,
    configureStdp,
    elapsedMs,
    resetTimer,
    deviceCount
) where

import Control.Monad (when, forM)
import Data.Array.MArray (newListArray)
import Data.Array.Storable (StorableArray, withStorableArray)
import Data.Bits (setBit)
import Data.List (foldl')
import Data.Word (Word64)
import Foreign.C.Types
import Foreign.C.String (peekCString)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray)
import Foreign.Marshal.Utils (fromBool, toBool)
import Foreign.Ptr
import Foreign.Storable (peek)

import Simulation.CommonFFI
import Simulation.CUDA.Address

import Types (Time, Delay, Weight, Idx)

#include <libnemo.h>


{- Runtime data is managed on the CUDA-side in a single structure -}
data CuRT = CuRT


foreign import ccall unsafe "nemo_new_network"
    c_newNetwork
        :: CUInt  -- ^ set reverse matrix (bool)
        -> CUInt  -- ^ max read period
        -> IO (Ptr CuRT)

-- for debugging, specify partition size
foreign import ccall unsafe "nemo_new_network_"
    c_newNetwork_
        :: CUInt  -- ^ set reverse matrix (bool)
        -> CUInt  -- ^ max read period
        -> CUInt  -- ^ max partition size
        -> IO (Ptr CuRT)


allocateRuntime :: Maybe Int -> Bool -> Int -> IO (Ptr CuRT)
allocateRuntime psize usingStdp maxProbePeriod =
    maybe (c_newNetwork us pp) (c_newNetwork_ us pp . fromIntegral) psize
    where
        us = fromBool usingStdp
        pp = fromIntegral maxProbePeriod


foreign import ccall unsafe "nemo_add_neuron" c_addNeuron
    :: Ptr CuRT
    -> CUInt   -- ^ global neuron index
    -> CFloat  -- ^ a
    -> CFloat  -- ^ b
    -> CFloat  -- ^ c
    -> CFloat  -- ^ d
    -> CFloat  -- ^ u
    -> CFloat  -- ^ v
    -> CFloat  -- ^ sigma
    -> IO ()


addNeuron :: Ptr CuRT -> Int
    -> Double -> Double -> Double -> Double
    -> Double -> Double -> Double -> IO ()
addNeuron rt nidx a b c d u v sigma =
    c_addNeuron rt (fromIntegral nidx) (f a) (f b) (f c) (f d) (f u) (f v) (f sigma)
    where
        f = realToFrac


-- free the device, clear all memory in Sim
foreign import ccall unsafe "nemo_delete_network" freeRT :: Ptr CuRT -> IO ()


-------------------------------------------------------------------------------
-- Loading data
-------------------------------------------------------------------------------


{- | Force copy of data to device -}
foreign import ccall unsafe "nemo_start_simulation" startSimulation :: Ptr CuRT -> IO ()



foreign import ccall unsafe "nemo_add_synapses"
    c_addSynapses :: Ptr CuRT
                -> CUInt        -- ^ source neuron index
                -> Ptr CUInt    -- ^ target neuron indices
                -> Ptr CUInt    -- ^ synapse delays
                -> Ptr CFloat   -- ^ synapse weights
                -> Ptr CUChar   -- ^ per-synapse plasticity
                -> CSize        -- ^ synapses count for this neuron/delay pair
                -> IO ()


addSynapses :: Ptr CuRT
        -> Int
        -> Ptr CUInt    -- ^ target neuron indices
        -> Ptr CUInt    -- ^ synapse delays
        -> Ptr CFloat   -- ^ synapse weights
        -> Ptr CUChar   -- ^ per-synapse plasticity
        -> Int          -- ^ synapses count for this neuron/delay pair
        -> IO ()
addSynapses rt pre nbuf dbuf wbuf spbuf len =
    when (len > 0) $
    c_addSynapses rt
        (fromIntegral pre)
        nbuf dbuf wbuf spbuf
        (fromIntegral $! len)


foreign import ccall unsafe "nemo_get_synapses" c_getCMDRow
        :: Ptr CuRT
        -> CUInt            -- ^ source partition
        -> CUInt            -- ^ source neuron
        -> CUInt            -- ^ delay
        -> Ptr (Ptr CUInt)  -- ^ target partitions
        -> Ptr (Ptr CUInt)  -- ^ target neurons
        -> Ptr (Ptr CFloat) -- ^ synapse weights
        -> Ptr (Ptr CUChar) -- ^ synapse plasticity
        -> IO CSize         -- ^ length of returned array



{- | Get (possibly modified) synapses for a single neuron and delay -}
getCMDRow
    :: Ptr CuRT
    -> PartitionIdx             -- ^ source partition
    -> NeuronIdx                -- ^ source neuron
    -> Delay                    -- ^ delay
    -> IO [(DeviceIdx, Weight, Bool)] -- ^ synapses
getCMDRow rt sp sn d = do
    alloca $ \p_ptr -> do
    alloca $ \n_ptr -> do
    alloca $ \w_ptr -> do
    alloca $ \s_ptr -> do -- plasticity
    c_len <- c_getCMDRow rt
            (fromIntegral sp)
            (fromIntegral sn)
            (fromIntegral d)
            p_ptr n_ptr w_ptr s_ptr
    let len = fromIntegral c_len
    p_list <- peekWith fromIntegral len p_ptr
    n_list <- peekWith fromIntegral len n_ptr
    w_list <- peekWith realToFrac len w_ptr
    s_list <- peekWith toBool len s_ptr
    let didx = zip p_list n_list
    return $! zip3 didx w_list s_list
    where
        peekWith f len ptr = (return . map f) =<< peekArray len =<< peek ptr


-------------------------------------------------------------------------------
-- Kernel execution
-------------------------------------------------------------------------------

foreign import ccall unsafe "nemo_sync_simulation"
    syncSimulation :: Ptr CuRT -> IO ()


{- The 'c_step' function is marked as safe, even though nothing could be
 - further from the truth. Haskell conflates the issues of re-entrancy and
 - concurrency in the FFI annotations. A foreign function can only run in a
 - concurrent manner if it is marked as safe. The step function is *not*
 - re-entrant.  If it is called in a re-entrant way the program will most
 - likely crash. We do, however, require concurrency for performance reason (in
 - particular performing network communication while the simulation is
 - running); hence 'safe'. -}
foreign import ccall safe "nemo_step"
    c_step :: Ptr CuRT  -- ^ kernel runtime data
           -> Ptr CInt  -- ^ Neuron indices of neurons with forced firing
           -> CSize     -- ^ Number of neurons whose firing is forced this step
           -> IO CInt   -- ^ Kernel status

foreign import ccall unsafe "nemo_flush_firing_buffer"
    c_flushFiringBuffer :: Ptr CuRT -> IO ()


{- | Perform a single simulation step, while buffering firing data on the
 - device, rather than reading it back to the host -}
stepBuffering sim fstim = do
    let flen = length fstim
        fbounds = (0, flen-1)
    fsNIdxArr <- newListArray fbounds $ map fromIntegral fstim
    withStorableArray fsNIdxArr  $ \fsNIdxPtr -> do
    checkStatus sim =<< c_step sim fsNIdxPtr (fromIntegral flen)


stepNonBuffering sim fstim = do
    stepBuffering sim fstim
    -- the simulation always records firing, so we just flush the buffer after
    -- the fact to avoid overflow.
    c_flushFiringBuffer sim



foreign import ccall unsafe "nemo_read_firing"
    c_readFiring :: Ptr CuRT
        -> Ptr (Ptr CUInt) -- cycles
        -> Ptr (Ptr CUInt) -- neuron idx
        -> Ptr CUInt       -- number of fired neurons
        -> Ptr CUInt       -- number of cycles
        -> IO ()


{- Return both the length of firing as well as a compact list of firings in
 - device indices -}
readFiring :: Ptr CuRT -> IO (Int, [(Time, Idx)])
readFiring rt = do
    alloca $ \cyclesPtr -> do
    alloca $ \nidxPtr   -> do
    alloca $ \nfiredPtr -> do
    alloca $ \ncyclesPtr -> do
    c_readFiring rt cyclesPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- return . fromIntegral =<< peek nfiredPtr
    cycles <- peekArray nfired =<< peek cyclesPtr
    nidx <- peekArray nfired =<< peek nidxPtr
    ncycles <- peek ncyclesPtr
    return $! (fromIntegral ncycles, zipWith fired cycles nidx)
    where
        fired c n = (fromIntegral c, fromIntegral n)


{- | Only return the firing count. This should be much faster -}
readFiringCount :: Ptr CuRT -> IO Int
readFiringCount rt = do
    alloca $ \cyclesPtr  -> do
    alloca $ \nidxPtr    -> do
    alloca $ \nfiredPtr  -> do
    alloca $ \ncyclesPtr -> do
    c_readFiring rt cyclesPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- peek nfiredPtr
    return $! fromIntegral nfired



-------------------------------------------------------------------------------
-- STDP
-------------------------------------------------------------------------------

foreign import ccall unsafe "nemo_enable_stdp" c_enableStdp
    :: Ptr rt
    -> Ptr CFloat -> CSize
    -> Ptr CFloat -> CSize
    -> CFloat -> CFloat
    -> IO ()


foreign import ccall unsafe "nemo_apply_stdp"
    c_applyStdp :: Ptr CuRT -> CFloat -> IO ()


instance ForeignKernel CuRT CFloat where
    ffi_enable_stdp = c_enableStdp
    ffi_apply_stdp = c_applyStdp


-------------------------------------------------------------------------------
-- Reporting
-------------------------------------------------------------------------------


foreign import ccall unsafe "nemo_print_cycle_counters" printCycleCounters
    :: Ptr CuRT -> IO ()


foreign import ccall unsafe "nemo_strerror" c_errorString
    :: Ptr CuRT -> IO (Ptr CChar)

errorString :: Ptr CuRT -> IO String
errorString rt = peekCString =<< c_errorString rt


{- | Check libnemo return status and fail with an error message if appropriate -}
checkStatus :: Ptr CuRT -> CInt -> IO ()
checkStatus rt status = when (status /= 0) $ fail =<< errorString rt



-------------------------------------------------------------------------------
-- Timing
-------------------------------------------------------------------------------


foreign import ccall unsafe "nemo_elapsed_ms" c_elapsedMs :: Ptr CuRT -> IO CLong

elapsedMs rt = return . fromIntegral =<< c_elapsedMs rt

foreign import ccall unsafe "nemo_reset_timer" resetTimer :: Ptr CuRT -> IO ()


-- | Return number of unique devices
foreign import ccall unsafe "nemo_device_count" deviceCount :: CInt
