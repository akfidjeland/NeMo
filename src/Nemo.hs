{-# LANGUAGE CPP #-}

module Main where

import Network (PortID(PortNumber))
import System.IO (stderr)
import System.Environment (getArgs)

#if defined(BENCHMARK_ENABLED)
import Benchmark (runBenchmark)
#endif
#if defined(EXAMPLES_ENABLED)
import Examples.SmallworldRun (runSmallworldExample)
import Examples.Random1kRun (runRandom1kExample)
import Examples.RingRun (runRingExample)
#endif
import ExternalClient (runExternalClient)
import Options
import Protocol (defaultPort)
import Server (runServer)
import Simulation.STDP.Options (stdpOptions)
#if defined(TEST_ENABLED)
import Test.RunTests (runTests)
#endif
import Types (Duration(..))


data ServerOptions = ServerOptions {
        optPort :: PortID,
        optRepetition :: Duration
    }


serverOptions = OptionGroup "Server options" (ServerOptions defaultPort Forever) optionDescr

optionDescr = [
        Option [] ["port"]
            (ReqArg (\a o -> return o { optPort = PortNumber $ toEnum $ read a }) "INT")
            "port number for client/server communication",

        Option [] ["once"]
            (NoArg (\o -> return o { optRepetition = Once }))
            "serve only a single client before shutting down (default: disabled)"
    ]

runBackendServer args0 = do
    (args, commonOpts) <- startOptProcessing args0
    serverOpts  <- processOptGroup serverOptions args
    -- simOpts     <- processOptGroup (simOptions LocalBackends) args
    -- cudaOpts    <- processOptGroup cudaOptions args
    endOptProcessing args
    -- let verbose = optVerbose commonOpts
    -- TODO: pass in options from command line
    let reps = optRepetition serverOpts
        port = optPort serverOpts
    runServer reps stderr (defaults stdpOptions) port


data Command
    = Client | Server
    | Benchmark
    | Test
    | Smallworld | Random1k | Ring
    | Unknown



{- | Return command along with remaining command-line arguments -}
command :: [String] -> (Command, String, [String])
command [] = (Client, [], [])
command args@(h:t) = if option h then (Client, [], args) else (cmd h, h, t)
    where
        option ('-':_) = True
        option _ = False

        cmd "server" = Server
        cmd "test" = Test
        cmd "benchmark" = Benchmark
        cmd "smallworld" = Smallworld
        cmd "random1k" = Random1k
        cmd "ring" = Ring
        cmd _ = Unknown



main = do
    (cmd, cmdname, args) <- return . command =<< getArgs
    case cmd of
        Client -> runExternalClient
        Server -> runBackendServer args
#if defined(BENCHMARK_ENABLED)
        Benchmark -> runBenchmark args
#endif
#if defined(TEST_ENABLED)
        Test -> runTests args
#endif
#if defined(EXAMPLES_ENABLED)
        Smallworld -> runSmallworldExample args
        Ring -> runRingExample args
        Random1k -> runRandom1kExample
#endif
        _ -> error $ "unknown command " ++ cmdname
