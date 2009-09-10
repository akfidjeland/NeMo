module Examples.SmallworldRun (runSmallworldExample) where

import System.Console.GetOpt
import System.IO

import Examples.Smallworld (smallworld)
import Construction.Izhikevich
import NSim
import Simulation.FiringStimulus


-- Firing stimulus can be provided either as an explicit list (see main) or as
-- a function that can be sampled (in the IO monad).
-- TODO: provide an example of a randomised stimulus
-- singleStimulus t
--    | t == 500  =
--    | otherwise = return []



data Options = Options {
        optClusterCount :: Int,
        optClusterSize  :: Int,
        optSynapses     :: Int,
        optRewiring     :: Double,
        optScaling      :: FT,
        optMaxDelay     :: Int
    }

defaultOptions :: Options
defaultOptions = Options {
        optClusterCount = 10,
        optClusterSize  = 100,
        optSynapses     = 20,
        optRewiring     = 0.01,
        optScaling      = 30,
        optMaxDelay     = 20
    }

-- put these options before the other command-line options
options = [

        Option ['h'] ["help"]
            (NoArg (\o -> showHelp options >> return o ))
            "show command-line options",

        Option [] ["cluster-count"]
            (ReqArg (\a o -> return o { optClusterCount = read a }) "INT")
            "number of distinct clustrrs",

        Option [] ["cluster-size"]
            (ReqArg (\a o -> return o { optClusterSize = read a }) "INT")
            "cluster size",

        Option [] ["synapse-count"]
            (ReqArg (\a o -> return o { optSynapses = read a }) "INT")
            "number of synapses per neuron",

        Option ['p'] ["rewiring-prob"]
            (ReqArg (\a o -> return o { optRewiring = read a }) "FLOAT")
            "probability of long-distance rewiring",

        Option ['F'] ["scaling-factor"]
            (ReqArg (\a o -> return o { optScaling = read a }) "FLOAT")
            "synaptic scaling factor",

        Option ['D'] ["max-delay"]
            (ReqArg (\a o -> return o { optMaxDelay = read a }) "INT")
            "maximum synapse delay for excitatory connections"
    ]


showHelp options = do
    putStr $ usageInfo ("Small-world network options:\n") options



runSmallworldExample args = do
    let (actions, _, _) = getOpt RequireOrder options args
    opts <- foldl (>>=) (return defaultOptions) actions
    execute
        (smallworld
            (optClusterCount opts)
            (optClusterSize opts)
            (optSynapses opts)
            (optRewiring opts)
            (optMaxDelay opts)
            (optScaling opts)
            False)
        (FiringList [(0, [0])])
