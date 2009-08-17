module Test.Network.Client (tests, test_clientSim) where

import Control.Concurrent
import Control.Exception
import System.IO
import System.FilePath ((</>))
import System.Directory (createDirectoryIfMissing)
import Test.HUnit

import Construction.Construction
import Construction.Izhikevich
import Construction.Network
import Construction.Synapse
import Examples.Smallworld (smallworldOrig)
import Network.Protocol
import Network.Server
import Options (defaults)
import Simulation.Backend (initSim)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.FiringStimulus
import Simulation.Options
import Simulation.Run
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
    serverThread <- forkIO runServer
    yield
    let
        sim1 = \f -> runSim (simOpts CPU 4) net
                        fstim f (defaults cudaOptions) (defaults stdpOptions)
        sim2 = \f -> runSim (simOpts (RemoteHost "localhost" testPort) 4) net
                        fstim f (defaults cudaOptions) (defaults stdpOptions)
    compareSims sim1 sim2
    throwTo serverThread ThreadKilled
    where
        -- TODO: share this with several other places in the testsuite
        net = build 123456 $ smallworldOrig
        fstim = FiringList [(0, [1])]



testPort = defaultPort + 2


runServer :: IO ()
runServer = do
    let logdir = "testsuite" </> "log"
    createDirectoryIfMissing True logdir
    let logfile = logdir </> "TestClient.log"
    bracket (openFile logfile WriteMode) (hClose) $ \hdl -> do
    serveSimulation
        hdl
        (show $ testPort)
        False
        -- TODO: get probe etc, from host as well
        (\net dt -> initSim
                    (net :: Network (IzhNeuron FT) Static)
                    (simOpts CPU dt)
                    (defaults cudaOptions))

simOpts backend dt =
    (defaults $ simOptions AllBackends) {
        optDuration   = Until 1000,
        optBackend    = backend,
        optTempSubres = dt
    }
