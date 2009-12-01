{- | Common FFI interface for different backends -}

module Simulation.CommonFFI (
    ConfigureStdp, configureStdp,
    ApplyStdp, applyStdp)
where

import Control.Monad (when)
import Foreign.C.Types (CUInt)
import Foreign.Marshal.Array (withArray)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)

import Simulation.STDP (StdpConf(..), prefireWindow, postfireWindow)


-- TODO: turn this into a class for a foreign interface


{- Type of foreign function for configuring STDP -}
type ConfigureStdp rt float
    =  Ptr rt      -- ^ pointer to foreign data structure containing simulation runtime
    -> CUInt       -- ^ length of pre-fire part of STDP window
    -> CUInt       -- ^ length of post-fire part of STDP window
    -> Ptr float   -- ^ lookup-table values (dt -> float) for STDP function prefire,
    -> Ptr float   -- ^ lookup-table values (dt -> float) for STDP function postfire,
    -> float       -- ^ max weight: limit for excitatory synapses
    -> float       -- ^ min weight: limit for inhibitory synapses
    -> IO ()


configureStdp :: (Storable f, Fractional f) => ConfigureStdp rt f -> Ptr rt -> StdpConf -> IO ()
configureStdp c_enable_stdp rt conf =
    when (stdpEnabled conf) $ do
    withArray (map realToFrac $ prefire conf) $ \prefire_ptr -> do
    withArray (map realToFrac $ postfire conf) $ \postfire_ptr -> do
    c_enable_stdp rt
        (fromIntegral $ prefireWindow conf)
        (fromIntegral $ postfireWindow conf)
        prefire_ptr
        postfire_ptr
        (realToFrac $ stdpMaxWeight conf)
        (realToFrac $ stdpMinWeight conf)



type ApplyStdp rt float = Ptr rt -> float -> IO ()


applyStdp :: (Fractional f) => ApplyStdp rt f -> Ptr rt -> Double -> IO ()
applyStdp c_apply_stdp rt reward = c_apply_stdp rt $ realToFrac reward
