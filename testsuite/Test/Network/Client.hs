module Test.Network.Client (tests, test_clientSim) where

import Control.Concurrent
import Control.Exception
import Network (PortID(PortNumber))
import System.IO
import System.FilePath ((</>))
import System.Directory (createDirectoryIfMissing)
import Test.HUnit

import Construction.Construction (build)
import Examples.Smallworld (smallworldOrig)
import Options (defaults)
import Server (runServer)
import Simulation.Backend (initSim)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.FiringStimulus
import Simulation.Options
import Simulation.Run (runSim)
import Simulation.STDP.Options (stdpOptions)
import Types

import Test.Comparative (compareSims)
import Test.Files

tests = test_clientSim


{- A simulation should have the same result whether it's run locally or over a
 - socket connection. This will only work on localhost, as otherwise small
 - hardware differences will break binary equivalence. -}
test_clientSim :: Test
test_clientSim = TestLabel "comparing client/server with local" $ TestCase $ do
    let logdir = "testsuite" </> "log"
        logfile = logdir </> "TestClient.log"
    createDirectoryIfMissing True logdir
    bracket (openFile logfile WriteMode) (hClose) $ \logTo -> do
    serverThread <- forkOS $ runServer Once logTo (simOpts CPU 4) (defaults stdpOptions) testPort
    yield
    let sim1 = \f -> runSim (simOpts CPU 4) net
                        fstim f (defaults cudaOptions) (defaults stdpOptions)
        sim2 = \f -> runSim (simOpts (RemoteHost "localhost" testPort) 4) net
                        fstim f (defaults cudaOptions) (defaults stdpOptions)
    compareSims sim1 sim2
    where
        -- TODO: share this with several other places in the testsuite
        net = build 123456 $ smallworldOrig
        fstim = FiringList [(0, [1])]

        testPort = PortNumber 56102

        simOpts backend dt =
            (defaults $ simOptions AllBackends) {
                optDuration   = Until 1000,
                optBackend    = backend,
                optTempSubres = dt
            }
