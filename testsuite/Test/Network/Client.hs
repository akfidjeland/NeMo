module Test.Network.Client (tests, test_clientSim) where

import Control.Concurrent
import Control.Exception (bracket)
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
import Options (defaultOptions)
import Simulation.Common
import Simulation.FiringStimulus
import Simulation.Run
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
        sim1 = \f -> runSim CPU duration net probeIdx probeF tempSubres
                        fstim f defaultOptions Nothing
        sim2 = \f -> runSim (RemoteHost "localhost" testPort) duration net
                        probeIdx probeF tempSubres fstim f defaultOptions Nothing
    compareSims sim1 sim2
    where
        -- TODO: share this with several other places in the testsuite
        net = build 123456 $ smallworldOrig
        tempSubres = 4
        duration = Until 1000
        probeIdx = All
        probeF = Firing :: ProbeFn IzhState
        fstim = FiringList [(0, [1])]


testPort = defaultPort + 1


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
        (\net tr -> initSim CPU
                    (net :: Network (IzhNeuron FT) Static)
                    All
                    (Firing :: ProbeFn IzhState)
                    tr False defaultOptions)
