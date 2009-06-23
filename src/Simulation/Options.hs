{-# LANGUAGE CPP #-}

{- | Options for controlling simulation, including duration and the simulation
 - backend used -}

module Simulation.Options (
        SimulationOptions(..),
        simOptions,
        BackendOptions(..)
    ) where

import Options
import Simulation.Common (Backend(..))
import Network.Protocol (defaultPort)
import Types


data BackendOptions
        = AllBackends
        | LocalBackends -- ^ all except remote (don't forward connections)
    deriving (Eq)

simOptions backends =
    OptionGroup "Simulation options" simDefaults $ simDescr backends


data SimulationOptions = SimulationOptions {
        optDuration   :: Duration,
        optTempSubres :: TemporalResolution,
        -- TODO: roll CUDA options into this
        optBackend    :: Backend
        -- TODO: roll STDP configuration into this
    }


simDefaults = SimulationOptions {
        optDuration   = Forever,
        optTempSubres = 4,
#if defined(CUDA_ENABLED)
        optBackend    = CUDA
#else
        optBackend    = CPU
#endif
    }


simDescr backend = local ++ if backend == AllBackends then remote else []
    where
        local = [

            Option ['t'] ["time"]    (ReqArg readDuration "INT")
                "duration of simulation in cycles (at 1ms resolution)",

            Option [] ["temporal-subresolution"]
                (ReqArg (\a o -> return o { optTempSubres = read a }) "INT")
                (withDefault (optTempSubres simDefaults)
                    "number of substeps per normal time step"),

#if defined(CUDA_ENABLED)
            Option [] ["gpu"]
                (NoArg (\o -> return o { optBackend=CUDA }))
                (withDefault ((==CUDA) $ optBackend simDefaults)
                    "use GPU backend for simulation, if present"),
#endif

            Option [] ["cpu"]
                (NoArg (\o -> return o { optBackend=CPU }))
                (withDefault ((==CPU) $ optBackend simDefaults)
                    "use CPU backend for simulation")
          ]

        remote = [
            Option [] ["remote"]
                (ReqArg getRemote "HOSTNAME[:PORT]")
                ("run simulation remotely on the specified server")
          ]


readDuration arg opt = return opt { optDuration = Until $ read arg }


-- format host:port
getRemote arg opts = return opts { optBackend = RemoteHost hostname port }
    where
        (hostname, port') = break (==':') arg
        port = if length port' > 1
            then read $ tail port'
            else defaultPort
