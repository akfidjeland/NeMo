{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Simulation.CUDA.KernelFFI (
    stepBuffering,
    stepNonBuffering,
    readFiring,
    applyStdp,
    setCMDRow,
    getCM,
    copyToDevice,
    deviceDiagnostics,
    syncSimulation,
    printCycleCounters,
    freeRT,
    CuRT,
    CMatrixIndex,
    cmatrixL0,
    cmatrixL1,
    -- TODO: hide details of this, instead move c ffi code into this module
    unCMatrixIndex,
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
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr
import Foreign.Storable (peek)

import Simulation.CommonFFI
import Simulation.CUDA.Address
import Simulation.CUDA.State (State(..), CuRT)

import Types (Time)

#include <kernel.h>



{- In the interface we manipulate/construct different connectivity matrices
 - using a numeric index -}
newtype CMatrixIndex = CMatrixIndex { unCMatrixIndex :: CSize }

cmatrixL0 :: CMatrixIndex
cmatrixL0 = CMatrixIndex #const CM_L0

cmatrixL1 :: CMatrixIndex
cmatrixL1 = CMatrixIndex #const CM_L1


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
                -> CSize        -- ^ matrix level: 0 or 1
                -> CUInt        -- ^ source partition index
                -> CUInt        -- ^ source neuron index
                -> CUInt        -- ^ synapse delay
                -> Ptr CFloat   -- ^ synapse weights
                -> Ptr CUInt    -- ^ target partition indices
                -> Ptr CUInt    -- ^ target neuron indices
                -> Ptr CUInt    -- ^ per-synapse plasticity
                -> CSize        -- ^ synapses count for this neuron/delay pair
                -> IO ()


setCMDRow rt wbuf pbuf nbuf spbuf level pre delay len =
    when (len > 0) $
    c_setCMDRow rt
        (unCMatrixIndex level)
        (fromIntegral $! partitionIdx pre)
        (fromIntegral $! neuronIdx pre)
        (fromIntegral delay)
        wbuf pbuf nbuf spbuf
        (fromIntegral $! len)



foreign import ccall unsafe "getCM"
    c_getCM :: Ptr CuRT
        -> CSize            -- ^ matrix level: 0 or 1
        -> Ptr (Ptr CInt)   -- ^ synapse target partitions
        -> Ptr (Ptr CInt)   -- ^ synapse target neurons
        -> Ptr (Ptr CFloat) -- ^ synapse weights
        -> Ptr CSize        -- ^ pitch of each row (synapses per delay)
        -> IO ()


getCM :: Ptr CuRT -> CMatrixIndex -> IO (Ptr CInt, Ptr CInt, Ptr CFloat, Int)
getCM rt lvl = do
    alloca $ \p_ptr   -> do
    alloca $ \n_ptr   -> do
    alloca $ \w_ptr   -> do
    alloca $ \len_ptr -> do
    c_getCM rt (unCMatrixIndex lvl) p_ptr n_ptr w_ptr len_ptr
    p   <- peek p_ptr
    n   <- peek n_ptr
    w   <- peek w_ptr
    len <- peek len_ptr
    return $! (p, n, w, fromIntegral len)



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
