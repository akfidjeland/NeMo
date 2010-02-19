{-# LANGUAGE ForeignFunctionInterface #-}

{- | Query device properties -}

module Simulation.CUDA.DeviceProperties (
    deviceCount
)where

import Foreign.C.Types

-- | Return number of unique devices
foreign import ccall unsafe "nemo_device_count" deviceCount :: CInt

