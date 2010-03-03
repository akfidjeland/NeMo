{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

{- | Common interface for simulator backends controlled via the FFI -}

module Simulation.CommonFFI (
    ForeignKernel(..),
    configureStdp,
    applyStdp
    )
where

import Control.Monad (when)
import Foreign.C.Types (CSize)
import Foreign.Marshal.Array (withArray)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)

import Simulation.STDP (StdpConf(..), prefireWindow, postfireWindow)


class (Fractional f, Storable f) => ForeignKernel rt f | rt -> f where

    ffi_enable_stdp
        :: Ptr rt -- ^ pointer to foreign data structure containing simulation runtime
        -> Ptr f  -- ^ lookup-table values (dt -> float) for STDP function prefire,
        -> CSize  -- ^ length of pre-fire part of STDP window
        -> Ptr f  -- ^ lookup-table values (dt -> float) for STDP function postfire,
        -> CSize  -- ^ length of post-fire part of STDP window
        -> f      -- ^ min weight: limit for inhibitory synapses
        -> f      -- ^ max weight: limit for excitatory synapses
        -> IO ()

    ffi_apply_stdp
        :: Ptr rt
        -> f      -- ^ reward
        -> IO ()



configureStdp :: ForeignKernel rt f => Ptr rt -> StdpConf -> IO ()
configureStdp rt conf =
    when (stdpEnabled conf) $ do
    withArray (map realToFrac $ prefire conf) $ \prefire_ptr -> do
    withArray (map realToFrac $ postfire conf) $ \postfire_ptr -> do
    ffi_enable_stdp rt
        prefire_ptr
        (fromIntegral $ prefireWindow conf)
        postfire_ptr
        (fromIntegral $ postfireWindow conf)
        (realToFrac $ stdpMinWeight conf)
        (realToFrac $ stdpMaxWeight conf)


applyStdp :: (ForeignKernel rt f, Fractional f) => Ptr rt -> Double -> IO ()
applyStdp rt reward = ffi_apply_stdp rt $ realToFrac reward
