{-# LANGUAGE CPP #-}

module Test.Examples.Smallworld (tests) where

import Test.HUnit

import Examples.Smallworld (smallworldOrig)
import Simulation.FiringStimulus
import Simulation.Options (Backend(..))
import Types

import Test.Regression


defaultTest =
    RegressionTest {
        name     = "smallworld",
        dataFile = "smallworld",
        netGen   = smallworldOrig,
        fstim    = FiringList [(0, [1]), (1, [1])],
        backend  = CPU,
        cycles   = 1000,
        rngSeed  = 123456,
        stdp     = Nothing
    }


tests = [
#if defined(CUDA_ENABLED)
    defaultTest {
        name     = "smallworld-gpu-1000",
        dataFile = "smallworld-gpu-1000",
        backend  = CUDA
    },

    defaultTest {
        name     = "smallworld-gpu-1000-stdp",
        dataFile = "smallworld-gpu-1000-stdp",
        backend  = CUDA,
        stdp     = Just 50
    },
#endif

    defaultTest {
        name     = "smallworld-cpu-1000",
        dataFile = "smallworld-cpu-1000"
    }

  ]
