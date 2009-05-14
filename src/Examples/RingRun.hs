module Main where

import System.Console.GetOpt
import System.Environment (getArgs)
import System.IO

import Examples.Ring
import NSim
import Construction.Izhikevich
import Simulation.FiringStimulus

data Options = Options {
        optRingCount :: Int,
        optRingSize  :: Int,
        optRingDelay :: Int
    }

defaultOptions :: Options
defaultOptions = Options {
        optRingCount = 1,
        optRingSize  = 1000,
        optRingDelay = 1
    }

-- put these options before the other command-line options
options = [

        Option [] ["ring-count"]
            (ReqArg (\a o -> return o { optRingCount = read a }) "INT")
            "select number of distinct ring networks",

        Option [] ["ring-size"]
            (ReqArg (\a o -> return o { optRingSize = read a }) "INT")
            "select size of each distinct ring network",

        Option [] ["ring-delay"]
            (ReqArg (\a o -> return o { optRingDelay = read a }) "INT")
            "select the delay of each synapse in the ring"
    ]


nrings n sz d = cluster (replicate n $ ring sz d) []


main = do
    args <- getArgs
    let (actions, _, _) = getOpt RequireOrder options args
    opts <- foldl (>>=) (return defaultOptions) actions
    let sz = optRingSize opts
        n  = optRingCount opts
        d  = optRingDelay opts
        fstim = take n $ iterate (+sz) 999
    hPutStrLn stderr $ (show n) ++ " network of size " ++ (show sz)
    execute "ring"
        (nrings n sz d)
        (FiringList [(0, fstim)]) All
        (Firing :: ProbeFn IzhState)
