{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Simulation.CUDA.KernelFFI (
    -- TODO: remove
    -- errorString,
    -- configuration
    CConfig,
    newConfiguration,
    setCudaPartitionSize,
    setFiringBufferLength,
    configureStdp,
    deleteConfiguration,
    -- construction
    CNetwork,
    newNetwork,
    addNeuron,
    addSynapses,
    deleteNetwork,
    getFiringBufferLength,
    -- simulation
    CSim,
    newSimulation,
    stepBuffering,
    stepNonBuffering,
    readFiring,
    applyStdp,
    getSynapses,
    deleteSimulation,
    elapsedMs,
    resetTimer
) where

import Control.Monad (when, forM, liftM)
import Control.Exception (evaluate)
import Control.Parallel.Strategies (rnf)
import Data.Array.MArray (newListArray)
import Data.Array.Storable (StorableArray, withStorableArray)
import Data.Bits (setBit)
import Data.List (foldl', zip4)
import Data.Word (Word64)
import Foreign.C.Types
import Foreign.C.String (peekCString)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray, withArray)
import Foreign.Marshal.Utils (fromBool, toBool)
import Foreign.Ptr
import Foreign.Storable (peek)

import Simulation.CUDA.Address

import Types (Time, Delay, Weight, Idx)
import Simulation.STDP (StdpConf(..), prefireWindow, postfireWindow)

#include <nemo.h>


{- Most functions return an error status -}
-- TODO: make sure we specify FFI functions to return this correctly
type CStatus = CInt

data CNetwork = CNetwork
data CSim = CSim
data CConfig = CConfig




-------------------------------------------------------------------------------
-- Configuration
-------------------------------------------------------------------------------


foreign import ccall unsafe "nemo_new_configuration" newConfiguration :: IO (Ptr CConfig)

foreign import ccall unsafe "nemo_log_stdout" enableLogStdout :: Ptr CConfig -> IO ()

foreign import ccall unsafe "nemo_enable_stdp" c_enableStdp
    :: Ptr CConfig
    -> Ptr CFloat -> CSize
    -> Ptr CFloat -> CSize
    -> CFloat -> CFloat
    -> IO CStatus


-- TODO: either use just lists here or make STDP conf part of the binding library
configureStdp :: Ptr CConfig -> StdpConf -> IO ()
configureStdp conf stdp =
    when (stdpEnabled stdp) $ do
    withArray (map realToFrac $ prefire stdp)  $ \prefire_ptr  -> do
    withArray (map realToFrac $ postfire stdp) $ \postfire_ptr -> do
    configure $ c_enableStdp conf
        prefire_ptr
        (fromIntegral $ prefireWindow stdp)
        postfire_ptr
        (fromIntegral $ postfireWindow stdp)
        (realToFrac $ stdpMinWeight stdp)
        (realToFrac $ stdpMaxWeight stdp)


foreign import ccall unsafe "nemo_set_firing_buffer_length" c_setFiringBufferLength
    :: Ptr CConfig -> CUInt -> IO CStatus

setFiringBufferLength :: Ptr CConfig -> Int -> IO ()
setFiringBufferLength conf len = configure $ c_setFiringBufferLength conf $ fromIntegral len


foreign import ccall unsafe "nemo_get_firing_buffer_length" c_getFiringBufferLength
    :: Ptr CConfig -> Ptr CUInt -> IO CStatus

getFiringBufferLength :: Ptr CConfig -> IO Int
getFiringBufferLength conf = do
    alloca $ \ptr -> do
    configure $ c_getFiringBufferLength conf ptr
    return . fromIntegral =<< peek ptr


foreign import ccall unsafe "nemo_set_cuda_partition_size" c_setCudaPartitionSize
    :: Ptr CConfig -> CUInt -> IO CStatus

setCudaPartitionSize :: Ptr CConfig -> Int -> IO ()
setCudaPartitionSize conf size = configure $ c_setCudaPartitionSize conf $ fromIntegral size


foreign import ccall unsafe "nemo_delete_configuration" deleteConfiguration :: Ptr CConfig -> IO ()


-------------------------------------------------------------------------------
-- Network construction
-------------------------------------------------------------------------------

foreign import ccall unsafe "nemo_new_network" newNetwork :: IO (Ptr CNetwork)


foreign import ccall unsafe "nemo_add_neuron" c_addNeuron
    :: Ptr CNetwork
    -> CUInt   -- ^ global neuron index
    -> CFloat  -- ^ a
    -> CFloat  -- ^ b
    -> CFloat  -- ^ c
    -> CFloat  -- ^ d
    -> CFloat  -- ^ u
    -> CFloat  -- ^ v
    -> CFloat  -- ^ sigma
    -> IO CStatus


addNeuron :: Ptr CNetwork -> Int
    -> Double -> Double -> Double -> Double
    -> Double -> Double -> Double -> IO ()
addNeuron net nidx a b c d u v sigma =
    construct $ c_addNeuron net (fromIntegral nidx) (f a) (f b) (f c) (f d) (f u) (f v) (f sigma)
    where
        f = realToFrac


foreign import ccall unsafe "nemo_add_synapses"
    c_addSynapses :: Ptr CNetwork
                -> CUInt        -- ^ source neuron index
                -> Ptr CUInt    -- ^ target neuron indices
                -> Ptr CUInt    -- ^ synapse delays
                -> Ptr CFloat   -- ^ synapse weights
                -> Ptr CUChar   -- ^ per-synapse plasticity
                -> CSize        -- ^ synapses count for this neuron/delay pair
                -> IO CStatus


-- TODO: expose this with a list interface instead
addSynapses :: Ptr CNetwork
        -> Int          -- ^ source neuron index
        -> Ptr CUInt    -- ^ target neuron indices
        -> Ptr CUInt    -- ^ synapse delays
        -> Ptr CFloat   -- ^ synapse weights
        -> Ptr CUChar   -- ^ per-synapse plasticity
        -> Int          -- ^ synapses count for this neuron/delay pair
        -> IO ()
addSynapses net pre nbuf dbuf wbuf spbuf len = when (len > 0) $ construct $
        c_addSynapses net (fromIntegral pre) nbuf dbuf wbuf spbuf (fromIntegral $! len)


-- free the device, clear all memory in Sim
-- TODO: make this a ForeignPtr again, but allow manual free as well
foreign import ccall unsafe "nemo_delete_network" deleteNetwork :: Ptr CNetwork -> IO ()


-------------------------------------------------------------------------------
-- Simulation
-------------------------------------------------------------------------------


foreign import ccall unsafe "nemo_new_simulation" newSimulation ::
    Ptr CNetwork -> Ptr CConfig -> IO (Ptr CSim)


{- The 'c_step' function is marked as safe, even though nothing could be
 - further from the truth. Haskell conflates the issues of re-entrancy and
 - concurrency in the FFI annotations. A foreign function can only run in a
 - concurrent manner if it is marked as safe. The step function is *not*
 - re-entrant.  If it is called in a re-entrant way the program will most
 - likely crash. We do, however, require concurrency for performance reason (in
 - particular performing network communication while the simulation is
 - running); hence 'safe'. -}
-- foreign import ccall safe "nemo_step"
foreign import ccall unsafe "nemo_step"
    c_step :: Ptr CSim
           -> Ptr CInt  -- ^ Neuron indices of neurons with forced firing
           -> CSize     -- ^ Number of neurons whose firing is forced this step
           -> IO CInt   -- ^ Kernel status

foreign import ccall unsafe "nemo_flush_firing_buffer"
    c_flushFiringBuffer :: Ptr CSim -> IO ()


{- | Perform a single simulation step, while buffering firing data on the
 - device, rather than reading it back to the host -}
stepBuffering
        :: Ptr CSim
        -> [Int]    -- ^ indices of neurons which should be forced to fire this cycle
        -> IO ()
stepBuffering sim fstim = do
    let flen = length fstim
        fbounds = (0, flen-1)
    fstim_arr <- newListArray fbounds $ map fromIntegral fstim
    withStorableArray fstim_arr  $ \fstim_ptr -> do
    simulate $ c_step sim fstim_ptr (fromIntegral flen)


stepNonBuffering sim fstim = do
    stepBuffering sim fstim
    -- the simulation always records firing, so we just flush the buffer after
    -- the fact to avoid overflow.
    c_flushFiringBuffer sim



foreign import ccall unsafe "nemo_read_firing"
    c_readFiring
        :: Ptr CSim
        -> Ptr (Ptr CUInt) -- cycles
        -> Ptr (Ptr CUInt) -- neuron idx
        -> Ptr CUInt       -- number of fired neurons
        -> Ptr CUInt       -- number of cycles
        -> IO CStatus


{- Return both the length of firing as well as a compact list of firings in
 - device indices -}
readFiring :: Ptr CSim -> IO (Int, [(Time, Idx)])
readFiring sim = do
    alloca $ \cyclesPtr -> do
    alloca $ \nidxPtr   -> do
    alloca $ \nfiredPtr -> do
    alloca $ \ncyclesPtr -> do
    simulate $! c_readFiring sim cyclesPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- return . fromIntegral =<< peek nfiredPtr
    cycles <- peekArray nfired =<< peek cyclesPtr
    nidx <- peekArray nfired =<< peek nidxPtr
    ncycles <- peek ncyclesPtr
    let ret = (fromIntegral ncycles, zipWith fired cycles nidx)
    -- Make sure we have read all foreign data, as it becomes invalidated on
    -- the next call to c_readFiring.
    evaluate $! rnf ret
    return $! ret
    where
        fired c n = (fromIntegral c, fromIntegral n)


{- | Only return the firing count. This should be much faster -}
readFiringCount :: Ptr CSim -> IO Int
readFiringCount sim = do
    alloca $ \cyclesPtr  -> do
    alloca $ \nidxPtr    -> do
    alloca $ \nfiredPtr  -> do
    alloca $ \ncyclesPtr -> do
    simulate $ c_readFiring sim cyclesPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- peek nfiredPtr
    return $! fromIntegral nfired



foreign import ccall unsafe "nemo_get_synapses" c_getSynapses
        :: Ptr CSim
        -> CUInt            -- ^ source neuron
        -> Ptr (Ptr CUInt)  -- ^ target neurons
        -> Ptr (Ptr CUInt)  -- ^ conductance delays
        -> Ptr (Ptr CFloat) -- ^ synapse weights
        -> Ptr (Ptr CUChar) -- ^ synapse plasticity
        -> Ptr CSize        -- ^ length of returned array
        -> IO CStatus


{- | Get (possibly modified) synapses for a single neuron and delay -}
getSynapses
    :: Ptr CSim
    -> NeuronIdx -- ^ source neuron
    -> IO [(Idx, Delay, Weight, Bool)]
getSynapses sim src = do
    alloca $ \n_ptr -> do -- target neuron
    alloca $ \d_ptr -> do -- delay
    alloca $ \w_ptr -> do -- weights
    alloca $ \p_ptr -> do -- plasticity
    alloca $ \len_ptr -> do
    simulate $ c_getSynapses sim (fromIntegral src) n_ptr d_ptr w_ptr p_ptr len_ptr
    len <- return . fromIntegral =<< peek len_ptr
    n_list <- peekWith fromIntegral len n_ptr
    d_list <- peekWith fromIntegral len d_ptr
    w_list <- peekWith realToFrac len w_ptr
    p_list <- peekWith toBool len p_ptr
    return $! zip4 n_list d_list w_list p_list
    where
        peekWith f len ptr = (return . map f) =<< peekArray len =<< peek ptr


foreign import ccall unsafe "nemo_apply_stdp" c_applyStdp :: Ptr CSim -> CFloat -> IO CStatus

applyStdp :: Ptr CSim -> Double -> IO ()
applyStdp sim reward = simulate $ c_applyStdp sim $ realToFrac reward



foreign import ccall unsafe "nemo_delete_simulation" deleteSimulation :: Ptr CSim -> IO ()



-------------------------------------------------------------------------------
-- Reporting
-------------------------------------------------------------------------------

foreign import ccall unsafe "nemo_strerror" c_errorString :: IO (Ptr CChar)

errorString :: IO String
errorString = peekCString =<< c_errorString



configure :: IO CStatus -> IO ()
configure action = do
    status <- action
    when (status /= 0) $ fail =<< errorString

construct :: IO CStatus -> IO ()
construct action = do
    status <- action
    when (status /= 0) $ fail =<< errorString


simulate :: IO CStatus -> IO ()
simulate action = do
    status <- action
    when (status /= 0) $ fail =<< errorString


-------------------------------------------------------------------------------
-- Timing
-------------------------------------------------------------------------------


foreign import ccall unsafe "nemo_elapsed_wallclock" c_elapsedMs :: Ptr CSim -> IO CULong

elapsedMs rt = return . fromIntegral =<< c_elapsedMs rt

foreign import ccall unsafe "nemo_reset_timer" resetTimer :: Ptr CSim -> IO ()
