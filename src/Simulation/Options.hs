{-# LANGUAGE CPP #-}

{- | Options for controlling simulation, including duration and the simulation
 - backend used -}

module Simulation.Options (
        Backend(..),
        SimulationOptions(..),
        simOptions,
        BackendOptions(..),
        defaultBackend,
        enablePipelining
    ) where

import Network (PortID(PortNumber))

import Options
import Protocol (defaultPort)
import Types


data Backend
        = CPU                   -- ^ (multi-core) CPU
#if defined(CUDA_ENABLED)
        | CUDA                  -- ^ CUDA-enabled GPU
#endif
        | RemoteHost String PortID -- ^ some other machine on specified port
    deriving (Eq, Show)

instance Eq PortID
instance Show PortID

-- TODO: remove
defaultBackend :: Backend
#if defined(CUDA_ENABLED)
defaultBackend = CUDA
#else
defaultBackend = CPU
#endif

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
        optBackend    :: Backend,
        optPipelined  :: Bool
        -- TODO: roll STDP configuration into this
    }


enablePipelining :: SimulationOptions -> SimulationOptions
enablePipelining opts = opts { optPipelined = True }


simDefaults = SimulationOptions {
        optDuration   = Forever,
        optTempSubres = 4,
#if defined(CUDA_ENABLED)
        optBackend    = CUDA,
#else
        optBackend    = CPU,
#endif
        optPipelined  = False
    }


simDescr backend = local ++ if backend == AllBackends then remote else []
    where
        local = [

            -- TODO: specifying a duration only makes sense in batch mode
            Option ['t'] ["time"]    (ReqArg readDuration "INT")
                "duration of simulation in cycles (at 1ms resolution)",

            Option [] ["euler-steps"]
                (ReqArg (\a o -> return o { optTempSubres = read a }) "INT")
                (withDefault (optTempSubres simDefaults)
                    "number of euler method steps per full simulation cycle"),

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
            then PortNumber $ toEnum $ read $ tail port'
            else defaultPort
