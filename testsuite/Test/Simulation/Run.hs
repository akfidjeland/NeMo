{-# LANGUAGE ScopedTypeVariables #-}
module Test.Simulation.Run (tests) where

import Control.Exception (handle, IOException)
import System.Directory (removeFile)
import System.IO (hClose)
import System.Random (mkStdGen)
import Test.HUnit
import Test.QuickCheck (generate)

import Construction.Construction
import Construction.Izhikevich
import Construction.Network
import Construction.Synapse
import Examples.Smallworld (smallworldOrig)
import Simulation.Common
import Options (defaults)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.FiringStimulus
import Simulation.FileSerialisation (encodeSimFile, decodeSimFile)
import Simulation.Options (simOptions, BackendOptions(..), optDuration)
import Simulation.Run
import Simulation.STDP.Options (stdpOptions)
import Types

import Test.Files
import Test.Comparative (comparisonTest)

tests = TestList [
        TestCase test_fileSerialisation,
        TestCase test_invalidDataDecoding,
        test_repeatedRun
    ]


-- | Network should be the same after a roundtrip to a file
test_fileSerialisation :: Assertion
test_fileSerialisation = do
    (f,h) <- openTemp "test_fileSerialisation"
    hClose h
    let net1 = generate (1000*20) (mkStdGen 0) (smallworldOrig) :: Network (IzhNeuron FT) Static
    encodeSimFile f net1 NoFiring
    (net2, _) <- decodeSimFile f
    removeFile f
    assertEqual "file serialisation/deserialisation" net1 net2


-- | We should get an exception (rather than stack overflow) if provided with
-- garbage data
test_invalidDataDecoding :: Assertion
test_invalidDataDecoding = do
    handle (\(_::IOException) -> return ()) $ do
    (net, _) <- decodeSimFile "/dev/random"
        :: IO (Network (IzhNeuron FT) Static, FiringStimulus)
    assertFailure "succeeded in decoding random data!?"
    -- would be surprising indeed. In practice we'd get a stack overflow


-- | We should get the same firing trace when running the same network twice.
-- This tests for the case where residual data from one run affects the next
test_repeatedRun :: Test
test_repeatedRun = comparisonTest sim sim "Two subsequent identical simulations"


-- TODO: factor this out and make it a default simulation somewhere
sim f = runSim simOpts net probeIdx probeF fstim f
            (defaults cudaOptions) (defaults stdpOptions)
    where
        -- TODO: factor this out and share with Test.Network.Client
        net = build 123456 $ smallworldOrig
        probeIdx = All
        probeF = Firing :: ProbeFn IzhState
        fstim = FiringList [(0, [1])]
        simOpts = (defaults $ simOptions LocalBackends) { optDuration = Until 1000 }
