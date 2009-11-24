module Test.Simulation.Run (tests) where

import Test.HUnit

import Construction.Construction
import Examples.Smallworld (smallworldOrig)
import Options (defaults)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.FiringStimulus (FiringStimulus(FiringList))
import Simulation.Options (simOptions, BackendOptions(..), optDuration)
import Simulation.Run
import Simulation.STDP.Options (stdpOptions)
import Types

import Test.Comparative (comparisonTest)

tests = TestList [
        test_repeatedRun
    ]



{- | We should get the same firing trace when running the same network twice.
 - This tests for the case where residual data from one run affects the next -}
test_repeatedRun :: Test
test_repeatedRun = comparisonTest sim sim "Two subsequent identical simulations"


-- TODO: factor this out and make it a default simulation somewhere
sim f = runSim simOpts net fstim f (defaults cudaOptions) (defaults stdpOptions)
    where
        -- TODO: factor this out and share with Test.Network.Client
        net = build 123456 $ smallworldOrig
        fstim = FiringList [(0, [1])]
        simOpts = (defaults $ simOptions LocalBackends) { optDuration = Until 1000 }
