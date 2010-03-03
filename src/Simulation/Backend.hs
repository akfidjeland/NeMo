{-# LANGUAGE CPP #-}

{- | Select and intialise one of several kinds of available simulation backends -}

module Simulation.Backend (
    initSim,
    module Simulation.Options
) where

import System.IO (hPutStrLn, stderr)

import Construction.Network (isEmpty)
import Simulation
import qualified Simulation.Remote as Remote (initSim)
import qualified Simulation.CPU as CPU (initSim)
#if defined(CUDA_ENABLED)
import qualified Simulation.CUDA as CUDA (initSim)
import Simulation.CUDA.Options
#endif
import Simulation.Options
import Simulation.STDP (stdpEnabled)


{- | Initialise simulation of the requested kind -}
initSim net simOpts cudaOpts stdpConf = do
    if isEmpty net
      then fail "network is empty"
      else case (optBackend simOpts) of
        -- TODO: add temporal resolution to CPU simulation
        CPU -> do
            return . BS =<< CPU.initSim net stdpConf
#if defined(CUDA_ENABLED)
        CUDA ->
            return . BS =<< CUDA.initSim (optPartitionSize cudaOpts) net stdpConf
#endif
        (RemoteHost hostname port) ->
            return . BS =<< Remote.initSim hostname port net simOpts stdpConf
