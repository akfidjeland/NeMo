{-# LANGUAGE CPP #-}

{- | Select and intialise one of several kinds of available simulation backends -}

module Simulation.Backend (initSim) where

import Control.Monad (when)
import System.IO (hPutStrLn, stderr)

import qualified Network.Client as Remote (initSim)
import Simulation
import qualified Simulation.CPU as CPU (initSim)
#if defined(CUDA_ENABLED)
import qualified Simulation.CUDA as CUDA (initSim, deviceCount)
import Simulation.CUDA.Options
#endif
import Simulation.Options
import Simulation.STDP (stdpEnabled)


{- | Initialise simulation of the requested kind -}
initSim net simOpts cudaOpts stdpConf = do
    backend <- chooseBackend $ optBackend simOpts
    case backend of
        -- TODO: add temporal resolution to CPU simulation
        CPU -> do
            when (stdpEnabled stdpConf) $ error "STDP not supported for CPU backend"
            return . BS =<< CPU.initSim net
#if defined(CUDA_ENABLED)
        CUDA ->
            return . BS =<< CUDA.initSim (optPartitionSize cudaOpts) net dt stdpConf
#endif
        (RemoteHost hostname port) ->
            return . BS =<< Remote.initSim hostname port net dt stdpConf
    where
        dt = optTempSubres simOpts



{- | Determine what backend to use, printing a warning to stderr if the chosen
 - one is not available. -}
chooseBackend
    :: Backend      -- ^ preferred backend
    -> IO Backend   -- ^ actual backend
chooseBackend CPU = return CPU
chooseBackend r@(RemoteHost _ _) = return r
#if defined(CUDA_ENABLED)
chooseBackend CUDA =
    if CUDA.deviceCount < 1
        then do
            hPutStrLn stderr "No CUDA-enabled devices found. Reverting to CPU simulation"
            return CPU
        else
            return CUDA
#endif


