{- | Wrapper for C-based simulation kernel -}

{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CPU.KernelFFI (RT, set, step, addSynapses, clear) where

import Control.Exception (assert)
import Control.Monad (when)
import Data.Array.Storable
import Foreign.C.Types
import Foreign.Marshal.Array (peekArray, withArrayLen)
import Foreign.Marshal.Utils (fromBool, toBool)
import Foreign.Ptr

import Types


{- Opaque handle to network stored in foreign code -}
data ForeignData = ForeignData

type RT = Ptr ForeignData

type CIdx = CUInt
#if defined(CPU_SINGLE_PRECISION)
type CFt = CFloat
#else
type CFt = CDouble
#endif
type CDelay = CUInt
type CWeight = CFt


set
    :: [Double] -- ^ a
    -> [Double] -- ^ b
    -> [Double] -- ^ c
    -> [Double] -- ^ d
    -> [Double] -- ^ u
    -> [Double] -- ^ v
    -> [Double] -- ^ sigma (0 if not input)
    -- TODO: remove need to pass in max delay
    -> Int      -- ^ max delay
    -> IO RT
set as bs cs ds us vs sigma maxDelay = do
    c_as <- newListArray bounds $ map realToFrac as
    c_bs <- newListArray bounds $ map realToFrac bs
    c_cs <- newListArray bounds $ map realToFrac cs
    c_ds <- newListArray bounds $ map realToFrac ds
    c_us <- newListArray bounds $ map realToFrac us
    c_vs <- newListArray bounds $ map realToFrac vs
    c_sigma <- newListArray bounds $ map realToFrac sigma
    withStorableArray c_as $ \as_ptr -> do
    withStorableArray c_bs $ \bs_ptr -> do
    withStorableArray c_cs $ \cs_ptr -> do
    withStorableArray c_ds $ \ds_ptr -> do
    withStorableArray c_us $ \us_ptr -> do
    withStorableArray c_vs $ \vs_ptr -> do
    withStorableArray c_sigma $ \sigma_ptr -> do
    c_set_network as_ptr bs_ptr cs_ptr ds_ptr us_ptr vs_ptr sigma_ptr c_sz c_maxDelay
    where
        sz = length as
        c_sz = fromIntegral sz
        c_maxDelay = fromIntegral maxDelay
        bounds = (0, sz-1)


foreign import ccall unsafe "cpu_set_network" c_set_network
    :: Ptr CFt     -- ^ a
    -> Ptr CFt     -- ^ b
    -> Ptr CFt     -- ^ c
    -> Ptr CFt     -- ^ d
    -> Ptr CFt     -- ^ u
    -> Ptr CFt     -- ^ v
    -> Ptr CFt     -- ^ sigma
    -> CSize       -- ^ network size
    -> CDelay      -- ^ max delay
    -> IO RT


foreign import ccall unsafe "cpu_add_synapses" c_add_synapses
    :: RT
    -> CIdx
    -> CDelay
    -> Ptr CIdx
    -> Ptr CWeight
    -> CSize
    -> IO ()

addSynapses :: RT -> Source -> Delay -> [Target] -> [Weight] -> IO ()
addSynapses rt src delay targets weights = do
    withArrayLen (map fromIntegral targets) $ \tlen tptr -> do
    withArrayLen (map realToFrac weights) $ \wlen wptr -> do
    assert (wlen == tlen) $ do
    when (wlen > 0) $ do
    c_add_synapses rt (c_int src) (c_int delay) tptr wptr (fromIntegral wlen)
    where
        c_int = fromIntegral


{- | Perform a single simulation step -}
step
    :: RT
    -> (Int, Int)               -- ^ neuron index bounds
    -> [Bool]                   -- ^ firing stimulus
    -> IO [Int]                 -- ^ indices of fired neurons
step rt bs fstim = do
    c_fstim <- newListArray bs $ map fromBool fstim
    withStorableArray c_fstim $ \fstim_ptr -> do
    fired <- peekArray sz =<< c_step rt fstim_ptr
    return $! map fst $ filter snd $ zip [0..] $ map toBool fired
    where
        sz = 1 + snd bs - fst bs


foreign import ccall unsafe "cpu_step" c_step
    :: RT
    -> Ptr CUInt       -- ^ boolean vector of firing stimulus
    -> IO (Ptr CUInt)  -- ^ boolean vector of fired neurons


foreign import ccall unsafe "cpu_delete_network" clear :: RT -> IO ()
