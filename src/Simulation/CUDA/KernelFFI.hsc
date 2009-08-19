{-# LANGUAGE CPP, ForeignFunctionInterface #-}

module Simulation.CUDA.KernelFFI (
    stepBuffering,
    applyStdp,
    setCMDRow,
    getCM,
    copyToDevice,
    deviceDiagnostics,
    syncSimulation,
    printCycleCounters,
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
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (withArray)
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr
import Foreign.Storable (peek)

import Simulation.CUDA.Address
import Simulation.CUDA.State (State(..), CuRT)
import Simulation.STDP (StdpConf(..),
            prefireWindow, postfireWindow,
            potentiationMask, depressionMask)

#include <kernel.h>



{- In the interface we manipulate/construction different connectivity matrices
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

-------------------------------------------------------------------------------
-- Kernel configuration
-------------------------------------------------------------------------------

maxPartitionSize :: Bool -> Int
maxPartitionSize = fromIntegral . c_maxPartitionSize . fromBool

foreign import ccall unsafe "maxPartitionSize" c_maxPartitionSize :: CInt -> CUInt


-------------------------------------------------------------------------------
-- Loading data
-------------------------------------------------------------------------------


{- | Force copy of data to device -}
foreign import ccall unsafe "copyToDevice" c_copyToDevice :: Ptr CuRT -> IO ()

copyToDevice rt = withForeignPtr rt c_copyToDevice


foreign import ccall unsafe "allocatedDeviceMemory"
    c_allocatedDeviceMemory :: Ptr CuRT -> IO CSize

deviceDiagnostics :: ForeignPtr CuRT -> IO String
deviceDiagnostics rt = do
    withForeignPtr rt $ \rtptr -> do
    dmem <- c_allocatedDeviceMemory rtptr
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
                    -> CSize        -- ^ synapses count for this neuron/delay pair
                    -> IO ()


setCMDRow rt wbuf pbuf nbuf level pre delay len =
    when (len > 0) $
    c_setCMDRow rt
        (unCMatrixIndex level)
        (fromIntegral $! partitionIdx pre)
        (fromIntegral $! neuronIdx pre)
        (fromIntegral delay)
        wbuf pbuf nbuf
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


foreign import ccall unsafe "step"
    c_step :: Ptr CuRT  -- ^ kernel runtime data
           -> CUShort   -- ^ cycle number (within current batch)
           -> CInt      -- ^ Sub-ms update steps
           -- External firing stimulus
           -> CSize     -- ^ Number of neurons whose firing is forced this step
           -> Ptr CInt  -- ^ Partition indices of neurons with forced firing
           -> Ptr CInt  -- ^ Neuron indices of neurons with forced firing
           -> IO CInt   -- ^ Kernel status


{- | Perform a single simulation step, while buffering firing data on the
 - device, rather than reading it back to the host -}
stepBuffering sim fstim currentCycle = do
    let flen = length fstim
        fbounds = (0, flen-1)
    fstim <- forM fstim $ rethrow "firing stimulus" $ deviceIdxM (att sim)
    fsPIdxArr <- newListArray fbounds (map (fromIntegral . partitionIdx) fstim)
    fsNIdxArr <- newListArray fbounds (map (fromIntegral . neuronIdx) fstim)
    withStorableArray fsPIdxArr  $ \fsPIdxPtr -> do
    withStorableArray fsNIdxArr  $ \fsNIdxPtr -> do
    withForeignPtr (rt sim)      $ \rtPtr     -> do
    kernelStatus <- c_step rtPtr currentCycle
        (dt sim) (fromIntegral flen) fsPIdxPtr fsNIdxPtr
    when (kernelStatus /= 0) $ fail "Backend error"
    where
        {- Run possibly failing computation, and propagate any errors with
         - additional prefix -}
        rethrow :: (Monad m) => String -> (a -> Either String b) -> a -> m b
        rethrow prefix f x = either (fail . (++) (prefix ++ ": ")) return (f x)



-------------------------------------------------------------------------------
-- STDP
-------------------------------------------------------------------------------


foreign import ccall unsafe "enableStdp" c_enableStdp
    :: Ptr CuRT
    -> CUInt       -- ^ length of pre-fire part of STDP window
    -> CUInt       -- ^ length of post-fire part of STDP window
    -> Word64      -- ^ bit-mask indicating the part of the STDP window which
                   --   is potentiation
    -> Word64      -- ^ bit-mask indicating the part of the STDP window which
                   --   is depression
    -> Ptr CFloat  -- ^ lookup-table values (dt -> float) for STDP function,
                   --   length: prefire_len + postfire_len
    -> CFloat      -- ^ max weight: limit for excitatory synapses
    -> IO ()


configureStdp :: ForeignPtr CuRT -> StdpConf -> IO ()
configureStdp rt conf =
    when (stdpEnabled conf) $ do
    withForeignPtr rt $ \rt_ptr -> do
    let lut = map realToFrac $ reverse (prefire conf) ++ (postfire conf)
    withArray lut $ \fnLut -> do
    c_enableStdp rt_ptr
        (fromIntegral $ prefireWindow conf)
        (fromIntegral $ postfireWindow conf)
        (word64 $ reverse $ potentiationMask conf)
        (word64 $ reverse $ depressionMask conf)
        fnLut
        (realToFrac $ stdpMaxWeight conf)

    where
        word64 :: [Bool] -> Word64
        word64 bits = foldl' set 0 $ zip bits [0..]

        set w (b, i) = if b then setBit w i else w



foreign import ccall unsafe "applyStdp"
    c_applyStdp :: Ptr CuRT -> CFloat -> IO ()


applyStdp :: Ptr CuRT -> Double -> IO ()
applyStdp rt reward = c_applyStdp rt $ realToFrac reward



-------------------------------------------------------------------------------
-- Reporting
-------------------------------------------------------------------------------

-- foreign import ccall unsafe "setVerbose" setVerbose :: IO ()

foreign import ccall unsafe "printCycleCounters" printCycleCounters
    :: Ptr CuRT -> IO ()



-------------------------------------------------------------------------------
-- Timing
-------------------------------------------------------------------------------


foreign import ccall unsafe "elapsedMs" c_elapsedMs :: Ptr CuRT -> IO CLong

elapsedMs rt = return . fromIntegral =<< c_elapsedMs rt

foreign import ccall unsafe "resetTimer" resetTimer :: Ptr CuRT -> IO ()
