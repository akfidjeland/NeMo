{-# LANGUAGE CPP #-}

{- Common simulation interface for different backends -}
module Simulation.Run (
    chooseBackend,
    initSim,
    runSim
) where

import Control.Monad (mapM, mapM_, when)
import Data.Maybe (isJust)
import System.IO (hPutStrLn, stderr)

import Simulation.FiringStimulus (firingStimulus)
import Simulation.Common
import qualified Simulation.CPU as CPU
#if defined(CUDA_ENABLED)
import qualified Simulation.CUDA as CUDA
import Simulation.CUDA.Options
#endif
import Simulation.Options
import Simulation.STDP
import qualified Network.Client as Remote (initSim)
import qualified Util.List as L (chunksOf)


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



-- | Get backend-specific step function
initSim simOpts net probeIdx probeF verbose cudaOpts stdpConf = do
    backend <- chooseBackend $ optBackend simOpts
    case backend of
        -- TODO: add temporal resolution to CPU simulation
        CPU -> do
            when (stdpEnabled stdpConf) $
                error "STDP not supported for CPU backend"
            CPU.initSim net probeIdx probeF
#if defined(CUDA_ENABLED)
        CUDA -> CUDA.initSim net probeIdx
            -- TODO: move option handling inside CUDA.hs
            (if optProbeDevice cudaOpts then Just probeF else Nothing)
            dt verbose
            (optPartitionSize cudaOpts)
            stdpConf
#endif
        (RemoteHost hostname port) ->
               Remote.initSim hostname port net dt stdpConf
    where
        dt = optTempSubres simOpts



-- | Run full simulation using the appropriate backend
runSim simOpts net probeIdx probeF fstimF outfn opts stdpConf = do
    -- TODO: don't pass stdp conf to init
    (Simulation sz run elapsed _ _ close) <-
        initSim simOpts net probeIdx probeF False opts stdpConf
    let cs = cycles duration
    fstim <- firingStimulus fstimF
    -- let stdp = applySTDP $ maybe Nothing stdpFrequency stdpConf
    let stdp = applySTDP stdpConf
        stim = zip fstim stdp
    mapM_ (aux run) $ L.chunksOf sz $ sample duration stim
    close
    where
        duration = optDuration simOpts
        aux run stim = do
            let (fstim, stdp) = unzip stim
            run fstim stdp >>= mapM_ outfn

        -- TODO: move to Simulation STDP?
        applySTDP conf = maybe
            (repeat STDPIgnore)
            (\freq -> cycle $ (replicate  (freq-1) STDPIgnore) ++ [STDPApply 1.0])
            (stdpFrequency conf)
