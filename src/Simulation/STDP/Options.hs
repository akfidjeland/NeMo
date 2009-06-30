{- | Command-line options controlling STDP -}
module Simulation.STDP.Options (stdpOptions) where

import Simulation.STDP
import Options

stdpOptions = OptionGroup "STDP options" stdpOptDefaults stdpOptDescr

stdpOptDefaults = STDPConf {
        stdpEnabled = False,
        stdpTauP = 20,
        stdpTauD = 20,
        stdpAlphaP = 1.0,
        stdpAlphaD = 0.8,
        -- TODO: may want to use Maybe here, and default to max in network
        stdpMaxWeight = 100.0, -- for no good reason
        stdpFrequency = Nothing
    }


stdpOptDescr = [
        Option [] ["stdp"]
            (NoArg (\o -> return o { stdpEnabled = True }))
            "Enable STDP",

        Option [] ["stdp-frequency"]
            (ReqArg (\a o -> return o { stdpFrequency = Just (read a) }) "INT")
             "frequency with which STDP should be applied (default: never)",

        Option [] ["stdp-a+"]
            (ReqArg (\a o -> return o { stdpAlphaP = read a }) "FLOAT")
            -- TODO: add back defaults
             -- (withDefault optStdpAlphaP "multiplier for synapse potentiation"),
             "multiplier for synapse potentiation",

        Option [] ["stdp-a-"]
            (ReqArg (\a o -> return o { stdpAlphaD = read a }) "FLOAT")
             --(withDefault optStdpAlphaD "multiplier for synapse depression"),
             "multiplier for synapse depression",

        Option [] ["stdp-t+"]
            (ReqArg (\a o -> return o { stdpTauP = read a }) "INT")
             -- withDefault optStdpTauP "Max temporal window for synapse potentiation"),
             "Max temporal window for synapse potentiation",

        Option [] ["stdp-t-"]
            (ReqArg (\a o -> return o { stdpTauD = read a }) "INT")
             -- (withDefault optStdpTauD "Max temporal window for synapse depression"),
             "Max temporal window for synapse depression",

        Option [] ["stdp-max-weight"]
            (ReqArg (\a o -> return o { stdpMaxWeight = read a }) "FLOAT")
             -- (withDefault optStdpMaxWeight "Set maximum weight for plastic synapses")
             "Set maximum weight for plastic synapses"
    ]
