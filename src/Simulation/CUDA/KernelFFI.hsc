{-# LANGUAGE CPP, ForeignFunctionInterface #-}

module Simulation.CUDA.KernelFFI (
    c_step,
    setCMDRow,
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
    enableSTDP,
    maxPartitionSize,
    elapsedMs,
    resetTimer
) where

import Control.Monad (when)
import Data.Array.Storable (StorableArray, withStorableArray)
import Foreign.C.Types
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr

import Simulation.CUDA.Address

#include <kernel.h>


{- Runtime data is managed on the CUDA-side in a single structure -}
data CuRT = CuRT


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




-------------------------------------------------------------------------------
-- Kernel execution
-------------------------------------------------------------------------------

foreign import ccall unsafe "syncSimulation"
    syncSimulation :: Ptr CuRT -> IO ()


foreign import ccall unsafe "step"
    c_step :: CUShort          -- ^ cycle number (within current batch)
           -> CInt             -- ^ Sub-ms update steps
           -> CInt             -- ^ Apply STDP? Boolean
           -> CFloat           -- ^ STDP reward
           -- External firing stimulus
           -> CSize            -- ^ Number of neurons whose firing is forced this step
           -> Ptr CInt         -- ^ Partition indices of neurons with forced firing
           -> Ptr CInt         -- ^ Neuron indices of neurons with forced firing
           -- Network state
           -> Ptr CuRT         -- ^ Kernel runtime data
           -> IO CInt          -- ^ Kernel status


-------------------------------------------------------------------------------
-- STDP
-------------------------------------------------------------------------------

foreign import ccall unsafe "enableSTDP" c_enableSTDP
    :: Ptr CuRT
    -> CInt     -- ^ tau_p : maximum time for potentiation
    -> CInt     -- ^ tau_d : maximum time for depression
    -> CFloat   -- ^ alpha_p: multiplier for potentiation
    -> CFloat   -- ^ alpha_d: multiplier for depression
    -> CFloat   -- ^ max weight: limit for excitatory synapses
    -> IO ()

enableSTDP rt tauP tauD alphaP alphaD maxWeight =
    withForeignPtr rt $ \rtptr ->
    c_enableSTDP rtptr
        (fromIntegral tauP) (fromIntegral tauD)
        (realToFrac alphaP) (realToFrac alphaD)
        (realToFrac maxWeight)



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
