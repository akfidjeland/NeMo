{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Simulation.CUDA.KernelFFI (
    stepBuffering,
    stepNonBuffering,
    readFiring,
    applyStdp,
    setCMDRow,
    getCMDRow,
    copyToDevice,
    deviceDiagnostics,
    syncSimulation,
    printCycleCounters,
    freeRT,
    CuRT,
    addNeuron,
    loadA, loadB, loadC, loadD,
    loadU, loadV,
    loadThalamicInputSigma,
    configureStdp,
    maxPartitionSize,
    elapsedMs,
    resetTimer
) where

import Control.Monad (when, forM)
import Data.Array.MArray (newListArray)
import Data.Array.Storable (StorableArray, withStorableArray)
import Data.Bits (setBit)
import Data.List (foldl')
import Data.Word (Word64)
import Foreign.C.Types
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray)
import Foreign.Marshal.Utils (fromBool, toBool)
import Foreign.Ptr
import Foreign.Storable (peek)

import Simulation.CommonFFI
import Simulation.CUDA.Address
import Simulation.CUDA.State (State(..), CuRT)

import Types (Time, Delay, Weight)

#include <kernel.h>


foreign import ccall unsafe "loadParam" c_loadParam
    :: Ptr CuRT
    -> CSize      -- ^ parameter vector index
    -> CSize      -- ^ partition index
    -> CSize      -- ^ partition size
    -> Ptr CFloat -- ^ data vector
    -> IO ()


loadParam :: CSize -> Ptr CuRT -> CSize -> Int -> StorableArray i CFloat -> IO ()
loadParam nvecIdx rt pidx len arr = do
    withStorableArray arr $ \ptr -> do
    c_loadParam rt nvecIdx pidx (fromIntegral len) ptr


loadA = loadParam #const PARAM_A
loadB = loadParam #const PARAM_B
loadC = loadParam #const PARAM_C
loadD = loadParam #const PARAM_D
loadU = loadParam #const STATE_U
loadV = loadParam #const STATE_V


foreign import ccall unsafe "addNeuron" c_addNeuron
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


-- TODO: remove this: it's no longer needed
foreign import ccall unsafe "loadThalamicInputSigma" c_loadThalamicInputSigma
    :: Ptr CuRT
    -> CSize      -- ^ partition index
    -> CSize      -- ^ partition size
    -> Ptr CFloat -- ^ data vector
    -> IO ()


loadThalamicInputSigma :: Ptr CuRT -> CSize -> Int -> StorableArray i CFloat -> IO ()
loadThalamicInputSigma rt pidx len arr = do
    withStorableArray arr $ \ptr -> do
    c_loadThalamicInputSigma rt pidx (fromIntegral len) ptr


-- free the device, clear all memory in Sim
foreign import ccall unsafe "freeRuntimeData" freeRT :: Ptr CuRT -> IO ()


-------------------------------------------------------------------------------
-- Kernel configuration
-------------------------------------------------------------------------------

maxPartitionSize :: Int
maxPartitionSize = #const MAX_PARTITION_SIZE


-------------------------------------------------------------------------------
-- Loading data
-------------------------------------------------------------------------------


{- | Force copy of data to device -}
foreign import ccall unsafe "copyToDevice" copyToDevice :: Ptr CuRT -> IO ()


foreign import ccall unsafe "allocatedDeviceMemory"
    c_allocatedDeviceMemory :: Ptr CuRT -> IO CSize

deviceDiagnostics :: Ptr CuRT -> IO String
deviceDiagnostics rt = do
    dmem <- c_allocatedDeviceMemory rt
    return $ "Allocated device memory: " ++ show dmem ++ "B"


foreign import ccall unsafe "setCMDRow"
    c_setCMDRow :: Ptr CuRT
                -> CUInt        -- ^ source neuron index
                -> CUInt        -- ^ synapse delay
                -> Ptr CUInt    -- ^ target neuron indices
                -> Ptr CFloat   -- ^ synapse weights
                -> Ptr CUChar   -- ^ per-synapse plasticity
                -> CSize        -- ^ synapses count for this neuron/delay pair
                -> IO ()


setCMDRow rt wbuf nbuf spbuf pre delay len =
    when (len > 0) $
    c_setCMDRow rt
        (fromIntegral pre)
        (fromIntegral delay)
        nbuf wbuf spbuf
        (fromIntegral $! len)



foreign import ccall unsafe "getCMDRow" c_getCMDRow
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

foreign import ccall unsafe "syncSimulation"
    syncSimulation :: Ptr CuRT -> IO ()


{- The 'c_step' function is marked as safe, even though nothing could be
 - further from the truth. Haskell conflates the issues of re-entrancy and
 - concurrency in the FFI annotations. A foreign function can only run in a
 - concurrent manner if it is marked as safe. The step function is *not*
 - re-entrant.  If it is called in a re-entrant way the program will most
 - likely crash. We do, however, require concurrency for performance reason (in
 - particular performing network communication while the simulation is
 - running); hence 'safe'. -}
foreign import ccall safe "step"
    c_step :: Ptr CuRT  -- ^ kernel runtime data
           -> CInt      -- ^ Sub-ms update steps
           -- External firing stimulus
           -> CSize     -- ^ Number of neurons whose firing is forced this step
           -> Ptr CInt  -- ^ Partition indices of neurons with forced firing
           -> Ptr CInt  -- ^ Neuron indices of neurons with forced firing
           -> IO CInt   -- ^ Kernel status

foreign import ccall unsafe "flushFiringBuffer"
    c_flushFiringBuffer :: Ptr CuRT -> IO ()


{- | Perform a single simulation step, while buffering firing data on the
 - device, rather than reading it back to the host -}
stepBuffering sim fstim = do
    let flen = length fstim
        fbounds = (0, flen-1)
    fstim <- forM fstim $ rethrow "firing stimulus" $ deviceIdxM (att sim)
    fsPIdxArr <- newListArray fbounds (map (fromIntegral . partitionIdx) fstim)
    fsNIdxArr <- newListArray fbounds (map (fromIntegral . neuronIdx) fstim)
    withStorableArray fsPIdxArr  $ \fsPIdxPtr -> do
    withStorableArray fsNIdxArr  $ \fsNIdxPtr -> do
    kernelStatus <- c_step (rt sim) (dt sim) (fromIntegral flen) fsPIdxPtr fsNIdxPtr
    when (kernelStatus /= 0) $ fail "Backend error"
    where
        {- Run possibly failing computation, and propagate any errors with
         - additional prefix -}
        rethrow :: (Monad m) => String -> (a -> Either String b) -> a -> m b
        rethrow prefix f x = either (fail . (++) (prefix ++ ": ")) return (f x)


stepNonBuffering sim fstim = do
    stepBuffering sim fstim
    -- the simulation always records firing, so we just flush the buffer after
    -- the fact to avoid overflow.
    c_flushFiringBuffer (rt sim)



foreign import ccall unsafe "readFiring"
    c_readFiring :: Ptr CuRT
        -> Ptr (Ptr CUInt) -- cycles
        -> Ptr (Ptr CUInt) -- partition idx
        -> Ptr (Ptr CUInt) -- neuron idx
        -> Ptr CUInt       -- number of fired neurons
        -> Ptr CUInt       -- number of cycles
        -> IO ()


{- Return both the length of firing as well as a compact list of firings in
 - device indices -}
readFiring :: Ptr CuRT -> IO (Int, [(Time, DeviceIdx)])
readFiring rt = do
    alloca $ \cyclesPtr -> do
    alloca $ \pidxPtr   -> do
    alloca $ \nidxPtr   -> do
    alloca $ \nfiredPtr -> do
    alloca $ \ncyclesPtr -> do
    c_readFiring rt cyclesPtr pidxPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- return . fromIntegral =<< peek nfiredPtr
    cycles <- peekArray nfired =<< peek cyclesPtr
    pidx <- peekArray nfired =<< peek pidxPtr
    nidx <- peekArray nfired =<< peek nidxPtr
    ncycles <- peek ncyclesPtr
    return $! (fromIntegral ncycles, zipWith3 fired cycles pidx nidx)
    where
        fired c p n = (fromIntegral c, (fromIntegral p, fromIntegral n))


{- | Only return the firing count. This should be much faster -}
readFiringCount :: Ptr CuRT -> IO Int
readFiringCount rt = do
    alloca $ \cyclesPtr  -> do
    alloca $ \pidxPtr    -> do
    alloca $ \nidxPtr    -> do
    alloca $ \nfiredPtr  -> do
    alloca $ \ncyclesPtr -> do
    c_readFiring rt cyclesPtr pidxPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- peek nfiredPtr
    return $! fromIntegral nfired



-------------------------------------------------------------------------------
-- STDP
-------------------------------------------------------------------------------

foreign import ccall unsafe "enableStdp"
    c_enableStdp :: Ptr rt -> CUInt -> CUInt
        -> Ptr CFloat -> Ptr CFloat -> CFloat -> CFloat -> IO ()


foreign import ccall unsafe "applyStdp"
    c_applyStdp :: Ptr CuRT -> CFloat -> IO ()


instance ForeignKernel CuRT CFloat where
    ffi_enable_stdp = c_enableStdp
    ffi_apply_stdp = c_applyStdp


-------------------------------------------------------------------------------
-- Reporting
-------------------------------------------------------------------------------


foreign import ccall unsafe "printCycleCounters" printCycleCounters
    :: Ptr CuRT -> IO ()


-------------------------------------------------------------------------------
-- Timing
-------------------------------------------------------------------------------


foreign import ccall unsafe "elapsedMs" c_elapsedMs :: Ptr CuRT -> IO CLong

elapsedMs rt = return . fromIntegral =<< c_elapsedMs rt

foreign import ccall unsafe "resetTimer" resetTimer :: Ptr CuRT -> IO ()
