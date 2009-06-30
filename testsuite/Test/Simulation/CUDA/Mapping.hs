module Test.Simulation.CUDA.Mapping (tests) where

import Test.HUnit

import Construction.Construction
import Construction.Izhikevich
import Examples.Smallworld
import Options (defaults)
import Simulation.CUDA.Options
import Simulation.Common
import Simulation.FiringStimulus
import Simulation.Options (SimulationOptions(..))
import Simulation.Run
import Simulation.STDP.Options (stdpOptions)
import Types

import Test.Comparative (comparisonTest)


{- Test that simulation results are unaffected by changes in cluster size used
 - on CUDA backend. Some of these also test that L1 connectivity works, as one
 - case uses L0 only, whereas the other uses both L0 and L1. -}
tests = TestList [
        testClusterSize 1000 1024,
        testClusterSize 1000 512,
        testClusterSize 1000 256,
        testClusterSize 1000 250
    ]


testClusterSize sz1 sz2 = comparisonTest (sim sz1) (sim sz2) lbl
    where
       lbl = "Comparing different partition sizes in the mapper: "
            ++ show sz1 ++ " vs " ++ show sz2


sim sz f = runSim (SimulationOptions duration dt CUDA) net probeIdx probeF fstim f
            ((defaults cudaOptions) { optPartitionSize = Just sz })
            (defaults stdpOptions)
    where
        -- TODO: factor this out and share with Test.Network.Client
        net = build 123456 $ smallworldOrig
        dt = 4
        duration = Until 1000
        probeIdx = All
        probeF = Firing :: ProbeFn IzhState
        fstim = FiringList [(0, [1])]
