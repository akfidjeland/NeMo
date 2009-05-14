{-# LANGUAGE ForeignFunctionInterface #-}

{- | Query device properties -}

module Simulation.CUDA.DeviceProperties (
    deviceCount
)where

import Foreign.Ptr
import Foreign.C.Types

-- opaque pointer to C-side data structure containing device properties
data DevPropStruct = DevPropStruct
type DevPropHandle = Ptr DevPropStruct


-- Get pointer to C-side data structure containing all the device properties
-- for the device with the given index
foreign import ccall unsafe "deviceProperties"
    deviceProperties :: Int -> IO DevPropHandle

-- | Return number of unique devices
foreign import ccall unsafe "deviceCount" deviceCount :: CInt

-- Return the number of bytes of global memory on the device
foreign import ccall unsafe "totalGlobalMem" totalGlobalMem :: DevPropHandle -> CInt

-- Return the number of bytes of shared memory per block
foreign import ccall unsafe "sharedMemPerBlock" sharedMemPerBlock :: DevPropHandle -> CInt

foreign import ccall unsafe "regsPerBlock" regsPerBlock :: DevPropHandle -> CInt

foreign import ccall unsafe "warpSize" warpSize :: DevPropHandle -> CInt

foreign import ccall unsafe "memPitch" memPitch :: DevPropHandle -> CInt

foreign import ccall unsafe "maxThreadsPerBlock" maxThreadsPerBlock :: DevPropHandle -> CInt

foreign import ccall unsafe "totalConstMem" totalConstMem :: DevPropHandle -> CInt

-- Return clock rate in kilohertz
foreign import ccall unsafe "clockRate" clockRate :: DevPropHandle -> CInt
