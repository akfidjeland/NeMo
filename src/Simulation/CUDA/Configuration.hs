{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CUDA.Configuration (
    configureKernel
) where

import Data.Array.Storable (withStorableArray)
import Data.Array.MArray (newListArray)
import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable (Storable)

import Simulation.CUDA.Mapping (CuNet, partitionSizes)

configureKernel :: CuNet n s -> IO ()
configureKernel = configurePartitionSize

-- Set array of configuration flags
setConfigArray :: (Storable b) =>
    [a] -> (a -> b) -> (CSize -> Ptr b -> IO ()) -> IO ()
setConfigArray flags cast configure = do
    h_flags <- newListArray (0, length flags - 1) $ map cast flags
    withStorableArray h_flags $ configure $ fromIntegral $ length flags


foreign import ccall unsafe "configurePartitionSize"
    c_configurePartitionSize :: CSize -> Ptr CInt -> IO ()

configurePartitionSize :: CuNet n s -> IO ()
configurePartitionSize net =
    setConfigArray (partitionSizes net) fromIntegral c_configurePartitionSize
