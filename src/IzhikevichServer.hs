module Main where

import Network (PortID(PortNumber))
import System.IO (stderr)

import Options
import Protocol (defaultPort)
import Server (runServer, forever)
import Simulation.STDP.Options (stdpOptions)


data ServerOptions = ServerOptions {
        optPort :: PortID
    }


serverOptions = OptionGroup "Server options" (ServerOptions defaultPort) optionDescr

optionDescr = [
        Option [] ["port"]
            (ReqArg (\a o -> return o { optPort = PortNumber $ toEnum $ read a }) "INT")
            "port number for client/server communication"
    ]

main = do
    (args, commonOpts) <- startOptProcessing
    serverOpts  <- processOptGroup serverOptions args
    -- simOpts     <- processOptGroup (simOptions LocalBackends) args
    -- cudaOpts    <- processOptGroup cudaOptions args
    endOptProcessing args
    -- let verbose = optVerbose commonOpts
    -- TODO: pass in options from command line
    runServer forever stderr (defaults stdpOptions) (optPort serverOpts)
