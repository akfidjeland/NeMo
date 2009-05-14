module Test.Examples.Random1k (tests) where

import Test.HUnit

import Examples.Random1k (random1k)
import Simulation.FiringStimulus
import Simulation.Common hiding (cycles)
import Types

import Test.Regression


tests = [
    RegressionTest {
        name     = "random1k-cpu-100",
        dataFile = "random1k-cpu-100",
        netGen   = random1k 800 200,
        fstim    = NoFiring,
        probe    = All,
        probeFn  = Firing,
        backend  = CPU,
        cycles   = 100,
        rngSeed  = 12345,
        stdp     = Nothing
    }
  ]
