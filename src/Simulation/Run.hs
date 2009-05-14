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

#if defined(CUDA_ENABLED)
import Options (optCuPartitionSz, optCuProbeDevice)
#endif
import Simulation.FiringStimulus (firingStimulus)
import Simulation.Common
import qualified Simulation.CPU as CPU
#if defined(CUDA_ENABLED)
import qualified Simulation.CUDA as CUDA
#endif
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
initSim reqBackend net probeIdx probeF dt verbose opts stdpConf = do
    backend <- chooseBackend reqBackend
    case backend of
        -- TODO: add temporal resolution to CPU simulation
        CPU -> do
            when (isJust stdpConf) $
                error "STDP not supported for CPU backend"
            CPU.initSim net probeIdx probeF
#if defined(CUDA_ENABLED)
        CUDA -> CUDA.initSim net probeIdx
            (if optCuProbeDevice opts then Just probeF else Nothing)
            dt verbose
            (optCuPartitionSz opts)
            stdpConf
#endif
        (RemoteHost hostname port) ->
               Remote.initSim hostname port net dt stdpConf



-- | Run full simulation using the appropriate backend
runSim backend duration net probeIdx probeF dt fstimF outfn opts stdpConf = do
    -- TODO: don't pass stdp conf to init
    (Simulation sz run elapsed _ close) <-
        initSim backend net probeIdx probeF dt False opts stdpConf
    let cs = cycles duration
    fstim <- firingStimulus fstimF
    let stdp = applySTDP $ maybe Nothing stdpFrequency stdpConf
        stim = zip fstim stdp
    putStrLn $ "len: " ++ (show $ length $ L.chunksOf sz $ sample duration stim)
    mapM_ (aux run) $ L.chunksOf sz $ sample duration stim
    close
    where
        aux run stim = do
            let (fstim, stdp) = unzip stim
            run fstim stdp >>= mapM_ outfn

        applySTDP Nothing = repeat STDPIgnore
        applySTDP (Just freq) = cycle $ (replicate (freq-1) STDPIgnore) ++ [STDPApply 1.0]
