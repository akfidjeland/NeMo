module Main where

import System.IO (stdout)

import Construction.Izhikevich (IzhNeuron, IzhState)
import Construction.Network (Network)
import Construction.Synapse (Static)
import Network.Protocol (defaultPort)
import Network.Server (serveSimulation)
import Simulation.Run (chooseBackend, initSim)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.Options (simOptions, optBackend, BackendOptions(..))
import Options
import Types



data ServerOptions = ServerOptions {
        optPort :: Int
    }

serverOptions = OptionGroup "Server options" (ServerOptions defaultPort) optionDescr

optionDescr = [
        Option [] ["port"]
            (ReqArg (\a o -> return o { optPort = read a }) "INT")
            "port number for client/server communication"
    ]

main = do
    (args, commonOpts) <- startOptProcessing
    serverOpts  <- processOptGroup serverOptions args
    simOpts     <- processOptGroup (simOptions ServerBackends) args
    cudaOpts    <- processOptGroup cudaOptions args
    endOptProcessing args
    backend <- chooseBackend $ optBackend simOpts
    let verbose = optVerbose commonOpts
    serveSimulation
        stdout
        (show $ optPort serverOpts)
        verbose
        -- TODO: get probe etc, from host as well
        (\net tr stdp -> initSim simOpts
                    (net :: Network (IzhNeuron FT) Static)
                    All
                    (Firing :: ProbeFn IzhState)
                    verbose cudaOpts stdp)
