-- | Load network (and stimulus) from file and run it
module Main where

import Data.Maybe (fromMaybe)
import System.Environment (getArgs)

import Construction.Network
import Construction.Izhikevich
import Construction.Synapse
import NSim
import Options
import Simulation.FiringStimulus
import Simulation.FileSerialisation (decodeSimFile)


type Net = Network (IzhNeuron FT) Static




main = do
    opts <- getOptions "nsim-run" defaultOptions $
            commonOptions ++
            loadOptions ++
            backendOptions ++
            simOptions
    -- let filename = optLoadNet opts
    let filename = fromMaybe (error "no file specified") $ optLoadNet opts
    -- let filename = "testsuite/misc/smallworld.dat"
    putStrLn $ "Loading file from " ++ filename
    (net, fstim) <- decodeSimFile filename :: IO (Net, FiringStimulus)
    execute "nsim-run" (return net) fstim All (Firing :: ProbeFn IzhState)
