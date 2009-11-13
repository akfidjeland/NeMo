{- | Wrapper for C-based simulation kernel -}

{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CPU.KernelFFI (
    RT,
    StimulusBuffer,
    newStimulusBuffer,
    set,
    step,
    readFiring,
    addSynapses,
    clear)
where

import Control.Applicative
import Control.Exception (assert)
import Control.Monad (when)
import Data.Array.Storable
import Foreign.C.Types
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray, withArrayLen)
import Foreign.Ptr
import Foreign.Storable (peek)

import Types (Source, Delay, Target, Weight)


{- Opaque handle to network stored in foreign code -}
data ForeignData = ForeignData

type RT = Ptr ForeignData

{- We pre-allocate a buffer to store firing stimulus. This is to avoid repeated
 - allocation. -}
type StimulusBuffer = StorableArray Int CUInt

newStimulusBuffer :: Int -> IO StimulusBuffer
newStimulusBuffer ncount = newListArray (0, ncount-1) $ repeat 0


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



type FiringStimulus = [Int]

{- | Perform a single simulation step, ignoring output. -}
step :: RT -> StimulusBuffer -> FiringStimulus -> IO ()
step rt c_fstim fstim = do
    c_deliver_spikes rt
    {- To avoid having to pass over the whole array of firing stimulus we just
     - flip the status of the ones which are affected this cycle. -}
    withElemsSet c_fstim fstim $ \arr -> withStorableArray arr (c_update rt)


readFiring :: RT -> IO [Int]
readFiring rt = do
    alloca $ \arr_ptr -> do
    alloca $ \len_ptr -> do
    c_read_firing rt arr_ptr len_ptr
    c_fired <- peek arr_ptr
    c_len <- peek len_ptr
    return . map fromIntegral =<< peekArray (fromIntegral c_len) c_fired


{- | Run computation with certain values of array set, then reset the array -}
withElemsSet :: (Ix i) => StorableArray i CUInt -> [i] -> (StorableArray i CUInt -> IO a) -> IO a
withElemsSet arr idx f = write 1 *> f arr <* write 0
    where
        write val = mapM_ (\i -> writeArray arr i val) idx



foreign import ccall unsafe "cpu_deliver_spikes" c_deliver_spikes :: RT -> IO ()


foreign import ccall unsafe "cpu_update" c_update
    :: RT
    -> Ptr CUInt       -- ^ boolean vector of firing stimulus
    -> IO ()


foreign import ccall unsafe "cpu_read_firing" c_read_firing
    :: RT
    -> Ptr (Ptr CUInt) -- ^ neuron indices of fired neurons
    -> Ptr (CUInt)     -- ^ number of fired neurons
    -> IO ()


foreign import ccall unsafe "cpu_delete_network" clear :: RT -> IO ()
