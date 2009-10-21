{- Network where neurons constantly fire. Running this network should flush out
 - buffer overflow errors related to firing -}

module Test.Examples.Dense (tests) where

import Construction
import Simulation.FiringStimulus
import Simulation.Options
import Types

import Test.Regression


tests = [
    RegressionTest {
        name     = "dense-gpu-1000",
        dataFile = "dense-gpu-1000",
        netGen   = dense 1000,
        fstim    = NoFiring,
        backend  = CUDA,
        cycles   = 1000,
        rngSeed  = 2000,
        stdp     = Nothing
    }
  ]



dense n = clusterN (replicate n (randomised exN)) [connect every nonself exS]
    where
        -- excitatory neuron
        exN r = mkNeuron2 0.02 b (v + 15*r^2) (8.0-6.0*r^2) u v thalamic
            where
                b = 0.2
                u = b * v
                v = -65.0
                thalamic = mkThalamic 5.0 r

        -- excitatory synapse
        exS = mkRSynapse (between 0.0 0.5) (fixed 1)
