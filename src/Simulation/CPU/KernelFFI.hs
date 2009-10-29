{- | Wrapper for C simulation kernel -}

{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CPU.KernelFFI (RT, set, update, clear) where

import Data.Array.Storable
import Foreign.C.Types
import Foreign.Marshal.Array (peekArray)
import Foreign.Marshal.Utils (fromBool, toBool)
import Foreign.Ptr



{- Opaque handle to network stored in foreign code -}
data ForeignData = ForeignData
type RT = Ptr ForeignData


set
    :: [Double] -- ^ a
    -> [Double] -- ^ b
    -> [Double] -- ^ c
    -> [Double] -- ^ d
    -> [Double] -- ^ u
    -> [Double] -- ^ v
    -> [Double] -- ^ sigma (0 if not input)
    -> IO RT
set as bs cs ds us vs sigma = do
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
    c_set_network as_ptr bs_ptr cs_ptr ds_ptr us_ptr vs_ptr sigma_ptr c_sz
    where
        sz = length as
        c_sz = fromIntegral sz
        bounds = (0, sz-1)


foreign import ccall unsafe "set_network" c_set_network
    :: Ptr CDouble -- ^ a
    -> Ptr CDouble -- ^ b
    -> Ptr CDouble -- ^ c
    -> Ptr CDouble -- ^ d
    -> Ptr CDouble -- ^ u
    -> Ptr CDouble -- ^ v
    -> Ptr CDouble -- ^ sigma
    -> CUInt      -- ^ network size
    -> IO RT



{- | Perform a single simulation step -}
update
    :: RT
    -> (Int, Int)               -- ^ neuron index bounds
    -- TODO: make this a storable array instead
    -> [Bool]                   -- ^ firing stimulus
    -> StorableArray Int Double -- ^ current stimulus
    -> IO [Int]                 -- ^ indices of fired neurons
update rt bs fstim c_current = do
    c_fstim <- newListArray bs $ map fromBool fstim
    withStorableArray c_current $ \current_ptr -> do
    withStorableArray c_fstim $ \fstim_ptr -> do
    fired <- peekArray sz =<< c_update rt fstim_ptr current_ptr
    return $! map fst $ filter snd $ zip [0..] $ map toBool fired
    where
        sz = 1 + snd bs - fst bs


foreign import ccall unsafe "update" c_update
    :: RT
    -> Ptr CUInt       -- ^ boolean vector of firing stimulus
    {- Note: this is slightly dodgy. We should really use CDouble here.
     - However, CDouble is just a newtype of Double, so doing a cast would be
     - wasteful -}
    -> Ptr Double      -- ^ input current for each neuron
    -> IO (Ptr CUInt)  -- ^ boolean vector of fired neurons


foreign import ccall unsafe "delete_network" clear :: RT -> IO ()
