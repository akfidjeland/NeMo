{- | Wrapper for C simulation kernel -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CPU.KernelFFI (RT, set, update, addSynapses, clear) where

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
type CFt = CDouble
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


foreign import ccall unsafe "set_network" c_set_network
    :: Ptr CDouble -- ^ a
    -> Ptr CDouble -- ^ b
    -> Ptr CDouble -- ^ c
    -> Ptr CDouble -- ^ d
    -> Ptr CDouble -- ^ u
    -> Ptr CDouble -- ^ v
    -> Ptr CDouble -- ^ sigma
    -> CUInt       -- ^ network size
    -> CDelay      -- ^ max delay
    -> IO RT


foreign import ccall unsafe "add_synapses" c_add_synapses
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
update
    :: RT
    -> (Int, Int)               -- ^ neuron index bounds
    -> [Bool]                   -- ^ firing stimulus
    -> IO [Int]                 -- ^ indices of fired neurons
update rt bs fstim = do
    c_fstim <- newListArray bs $ map fromBool fstim
    withStorableArray c_fstim $ \fstim_ptr -> do
    fired <- peekArray sz =<< c_update rt fstim_ptr
    return $! map fst $ filter snd $ zip [0..] $ map toBool fired
    where
        sz = 1 + snd bs - fst bs


foreign import ccall unsafe "update" c_update
    :: RT
    -> Ptr CUInt       -- ^ boolean vector of firing stimulus
    -> IO (Ptr CUInt)  -- ^ boolean vector of fired neurons


foreign import ccall unsafe "delete_network" clear :: RT -> IO ()
