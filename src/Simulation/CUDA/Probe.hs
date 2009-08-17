{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CUDA.Probe (readFiring, readFiringCount) where

import Data.Array.MArray (getElems)
import Data.Array.Storable (unsafeForeignPtrToStorableArray)
import Foreign.C.Types (CUInt, CSize)
import Foreign.ForeignPtr
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr
import Foreign.Storable (peek)

import Simulation.CUDA.Memory (CuRT)
import Simulation.CUDA.Address (DeviceIdx)
import Types (Time)


foreign import ccall unsafe "readFiring"
    c_readFiring :: Ptr CuRT
        -> Ptr (Ptr CUInt) -- cycles
        -> Ptr (Ptr CUInt) -- partition idx
        -> Ptr (Ptr CUInt) -- neuron idx
        -> Ptr CUInt       -- number of fired neurons
        -> Ptr CUInt       -- number of cycles
        -> IO ()


{- Return both the length of firing as well as a compact list of firings in
 - device indices -}
readFiring :: ForeignPtr CuRT -> IO (Int, [(Time, DeviceIdx)])
readFiring rtdata = do
    withForeignPtr rtdata $ \rtptr -> do
    alloca $ \cyclesPtr -> do
    alloca $ \pidxPtr   -> do
    alloca $ \nidxPtr   -> do
    alloca $ \nfiredPtr -> do
    alloca $ \ncyclesPtr -> do
    c_readFiring rtptr cyclesPtr pidxPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- peek nfiredPtr
    cycles <- readArr cyclesPtr nfired
    pidx <- readArr pidxPtr nfired
    nidx <- readArr nidxPtr nfired
    ncycles <- peek ncyclesPtr
    return $! (fromIntegral ncycles, zipWith3 fired cycles pidx nidx)
    where
        fired c p n = (fromIntegral c, (fromIntegral p, fromIntegral n))
        readArr ptr len = do
            fptr <- newForeignPtr_ =<< peek ptr
            getElems =<< unsafeForeignPtrToStorableArray fptr (0, fromIntegral len-1)


{- | Only return the firing count. This should be much faster -}
readFiringCount :: ForeignPtr CuRT -> IO Int
readFiringCount rtdata = do
    withForeignPtr rtdata $ \rtptr -> do
    alloca $ \cyclesPtr  -> do
    alloca $ \pidxPtr    -> do
    alloca $ \nidxPtr    -> do
    alloca $ \nfiredPtr  -> do
    alloca $ \ncyclesPtr -> do
    c_readFiring rtptr cyclesPtr pidxPtr nidxPtr nfiredPtr ncyclesPtr
    nfired <- peek nfiredPtr
    return $! fromIntegral nfired
