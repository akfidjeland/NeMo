{-# LANGUAGE CPP #-}

module Test.Examples.Ring (tests) where

import Control.Monad (when)
import Data.Maybe
import Data.IORef
import Test.HUnit

import Construction.Construction (build)
import Construction.Izhikevich
import Construction.Network (Network)
import Construction.Synapse (Static)
import Examples.Ring (ring)
import Options
import Simulation.CUDA.Options (cudaOptions, optPartitionSize)
import Simulation.FiringStimulus
import Simulation.Options (simOptions, optDuration, optBackend, BackendOptions(..), Backend(..))
import Simulation.Run (runSim)
import Simulation.STDP.Options (stdpOptions)
import Types



checkFiring :: Int -> Idx -> Int -> Time -> FiringOutput -> Assertion
checkFiring size impulse delay = \t (FiringOutput fs) -> do
    when (delay == 1 && length fs /= 1) $ assertFailure $
        "ring network did not produce exactly one firing during cycle "
        ++ (show t) ++ ", instead it produced " ++ (show fs)
    let expected = if t `mod` delay == 0
            then [(impulse + (t `div` delay)) `mod` size]
            else []
    assertEqual (mismatch_msg t) expected fs
    where
        mismatch_msg c = "ring network firing mismatch in cycle " ++ show c


testRing :: Int -> Idx -> Int -> Backend -> Maybe Int -> Assertion
testRing size impulse delay backend partitionSize = do
    let net = build 123456 $ ring size delay :: Network IzhNeuron Static
        fstim = FiringList [(0, [impulse])]
#if defined(CUDA_ENABLED)
        opts = (defaults cudaOptions) { optPartitionSize = partitionSize }
#else
        opts = defaults cudaOptions
#endif
    let test = checkFiring size impulse delay
    runSim simOpts net fstim test opts (defaults stdpOptions)
    where
        simOpts = (defaults $ simOptions LocalBackends) {
                optBackend = backend,
                optDuration = Until $ size * 2
            }



tests = TestList [
#if defined(CUDA_ENABLED)
        TestLabel "1000-sized ring on gpu"
           (TestCase $ testRing 1000 0 1 CUDA $ Just 1000),
        -- excercise 10 partitions
        TestLabel "1000-sized ring on gpu using partition size of 128"
           (TestCase $ testRing 1000 0 1 CUDA $ Just 128),

        -- excercise 40 partitions, requiring timeslicing of MPs
        TestLabel "4000-sized ring on gpu using partition size of 128, testing spikes crosing partition boundaries"
           (TestCase $ testRing 1000 0 1 CUDA $ Just 128),

        TestLabel "2000-sized ring on gpu with spike injection into partitions outside partition 0 (neuron 1500)"
           (TestCase $ testRing 2000 1500 1 CUDA $ Just 1000),

        TestLabel "2000-sized ring on gpu with spike injection into last neuron in partition 14 (of 16)"
           (TestCase $ testRing 2000 1792 1 CUDA $ Just 128),

        TestLabel "2000-sized ring on gpu using partition size of 128, with delays of 3"
           (TestCase $ testRing 2000 0 3 CUDA $ Just 128),
#endif
        TestLabel "1000-sized ring on cpu, delays of 1"
           (TestCase $ testRing 1000 0 1 CPU Nothing),

        TestLabel "1000-sized ring on cpu, delays of 3"
           (TestCase $ testRing 1000 0 3 CPU Nothing)
    ]
