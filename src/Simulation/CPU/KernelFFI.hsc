{- | Wrapper for C-based simulation kernel -}

{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CPU.KernelFFI (
    RT,
    StimulusBuffer,
    newStimulusBuffer,
    newNetwork,
    setNetwork,
    addNeuron,
    addSynapses,
    start,
    step,
    readFiring,
    elapsedMs,
    resetTimer,
    clear)
where

import Control.Applicative
import Control.Exception (assert)
import Control.Monad (when)
import Data.Array.Storable
import Foreign.C.Types
import Foreign.C.String (CString, peekCString)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (peekArray, withArrayLen)
import Foreign.Ptr
import Foreign.Storable (peek)

import Types (Source, Delay, Target, Weight)

#include <cpu_kernel.h>


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

type CStatus = CUInt


statusOK = #const STATUS_OK
statusError = #const STATUS_ERROR

{- Call kernel and trow an exception on error -}
callKernel_ :: RT -> IO CStatus -> IO ()
callKernel_ rt call = do
    status <- call
    when (status == statusError) $ throwError
    where
        throwError = do
            putStrLn $ "hs: caught error"
            str <- peekCString =<< c_last_error rt
            putStrLn $ "hs: failing"
            fail str



foreign import ccall unsafe "cpu_new_network" newNetwork :: IO RT


setNetwork
    :: [Double] -- ^ a
    -> [Double] -- ^ b
    -> [Double] -- ^ c
    -> [Double] -- ^ d
    -> [Double] -- ^ u
    -> [Double] -- ^ v
    -> [Double] -- ^ sigma (0 if not input)
    -> IO RT
setNetwork as bs cs ds us vs sigma = do
    rt <- newNetwork
    zipWithM8_ (addNeuron rt) [0..] as bs cs ds us vs sigma
    return rt
    where
        zipWithM8_ f a1 a2 a3 a4 a5 a6 a7 a8 =
            sequence_ (zipWith8 f a1 a2 a3 a4 a5 a6 a7 a8)
        zipWith8 z (a:as) (b:bs) (c:cs) (d:ds) (e:es) (f:fs) (g:gs) (h:hs)
                           =  z a b c d e f g h : zipWith8 z as bs cs ds es fs gs hs
        zipWith8 _ _ _ _ _ _ _ _ _ = []



foreign import ccall unsafe "cpu_set_network" c_set_network
    :: Ptr CFt     -- ^ a
    -> Ptr CFt     -- ^ b
    -> Ptr CFt     -- ^ c
    -> Ptr CFt     -- ^ d
    -> Ptr CFt     -- ^ u
    -> Ptr CFt     -- ^ v
    -> Ptr CFt     -- ^ sigma
    -> CSize       -- ^ network size
    -> IO RT


addNeuron
    :: RT -> Int
    -> Double -> Double -> Double -> Double
    -> Double -> Double -> Double
    -> IO ()
addNeuron rt idx a b c d u v s =
    callKernel_ rt $ c_add_neuron rt (fromIntegral idx) (cast a) (cast b) (cast c) (cast d) (cast u) (cast v) (cast s)
    where
        cast = realToFrac


foreign import ccall unsafe "cpu_add_neuron" c_add_neuron
    :: RT
    -> CIdx
    -> CFt -> CFt -> CFt -> CFt -- a, b, c, d
    -> CFt -> CFt -> CFt        -- u, v, sigma
    -> IO CStatus



addSynapses :: RT -> Source -> Delay -> [Target] -> [Weight] -> IO ()
addSynapses rt src delay targets weights = do
    withArrayLen (map fromIntegral targets) $ \tlen tptr -> do
    withArrayLen (map realToFrac weights) $ \wlen wptr -> do
    assert (wlen == tlen) $ do
    when (wlen > 0) $ do
    callKernel_ rt $ c_add_synapses rt (c_int src) (c_int delay) tptr wptr (fromIntegral wlen)
    where
        c_int = fromIntegral


foreign import ccall unsafe "cpu_add_synapses" c_add_synapses
    :: RT
    -> CIdx
    -> CDelay
    -> Ptr CIdx
    -> Ptr CWeight
    -> CSize
    -> IO CStatus


{- | Finalize construction and set up run-time data structures -}
start rt = callKernel_ rt $ c_start rt
foreign import ccall unsafe "cpu_start_simulation" c_start :: RT -> IO CStatus


type FiringStimulus = [Int]

{- | Perform a single simulation step, ignoring output. -}
step :: RT -> StimulusBuffer -> FiringStimulus -> IO ()
step rt c_fstim fstim = do
    c_deliver_spikes rt
    {- To avoid having to pass over the whole array of firing stimulus we just
     - flip the status of the ones which are affected this cycle. -}
    callKernel_ rt $ withElemsSet c_fstim fstim $ \arr -> withStorableArray arr (c_update rt)


readFiring :: RT -> IO [Int]
readFiring rt = do
    alloca $ \arr_ptr -> do
    alloca $ \len_ptr -> do
    c_read_firing rt arr_ptr len_ptr
    c_fired <- peek arr_ptr
    c_len <- peek len_ptr
    return . map fromIntegral =<< peekArray (fromIntegral c_len) c_fired


{- | Run computation with certain values of array set, then reset the array -}
withElemsSet
    :: (Ix i)
    => StorableArray i CUInt
    -> [i]
    -> (StorableArray i CUInt -> IO a)
    -> IO a
withElemsSet arr idx f = write 1 *> f arr <* write 0
    where
        write val = mapM_ (\i -> writeArray arr i val) idx



foreign import ccall unsafe "cpu_deliver_spikes" c_deliver_spikes
    :: RT -> IO CStatus


foreign import ccall unsafe "cpu_update" c_update
    :: RT
    -> Ptr CUInt       -- ^ boolean vector of firing stimulus
    -> IO CStatus


foreign import ccall unsafe "cpu_read_firing" c_read_firing
    :: RT
    -> Ptr (Ptr CUInt) -- ^ neuron indices of fired neurons
    -> Ptr (CUInt)     -- ^ number of fired neurons
    -> IO CStatus


foreign import ccall unsafe "cpu_delete_network" clear :: RT -> IO ()

resetTimer rt = callKernel_ rt $ c_reset_timer rt

foreign import ccall unsafe "cpu_reset_timer" c_reset_timer :: RT -> IO CStatus

foreign import ccall unsafe "cpu_elapsed_ms" c_elapsed_ms :: RT -> IO CLong

foreign import ccall unsafe "cpu_last_error" c_last_error :: RT -> IO CString

elapsedMs rt = return . fromIntegral =<< c_elapsed_ms rt
