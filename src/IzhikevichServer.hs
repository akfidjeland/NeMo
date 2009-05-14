module Main where

import System.IO (stdout)

import Construction.Izhikevich (IzhNeuron, IzhState)
import Construction.Network (Network)
import Construction.Synapse (Static)
import Network.Protocol (defaultPort)
import Network.Server (serveSimulation)
import Simulation.Run (chooseBackend, initSim)
import Options
import Types


main = do
    opts <- getOptions "nsim-server" defaultOptions $
            commonOptions ++
            verbosityOptions ++
            serverBackendOptions ++
            networkOptions
    backend <- chooseBackend (optBackend opts)
    let verbose = optVerbose opts
    -- bracket (openFile "/var/log/nsim.log" WriteMode) hClose $ \hdl -> do
    serveSimulation
        stdout
        (show $ optPort opts)
        verbose
        -- TODO: get probe etc, from host as well
        (\net tr stdp -> initSim backend
                    (net :: Network (IzhNeuron FT) Static)
                    All
                    (Firing :: ProbeFn IzhState)
                    tr verbose opts stdp)
