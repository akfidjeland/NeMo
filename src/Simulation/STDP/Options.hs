{- | Command-line options controlling STDP -}

module Simulation.STDP.Options (stdpOptions) where

import Simulation.STDP
import Options

stdpOptions = OptionGroup "STDP options" stdpOptDefaults stdpOptDescr

stdpOptDefaults = asymStdp { stdpEnabled = False }


stdpOptDescr = [
        Option [] ["stdp"]
            (NoArg (\o -> return o { stdpEnabled = True }))
            "Enable STDP",

        Option [] ["stdp-frequency"]
            (ReqArg (\a o -> return o { stdpFrequency = Just (read a) }) "INT")
             "frequency with which STDP should be applied (default: never)",

        Option [] ["stdp-max-weight"]
            (ReqArg (\a o -> return o { stdpMaxWeight = read a }) "FLOAT")
             (withDefault (stdpMaxWeight stdpOptDefaults)
                 "Set maximum weight for plastic synapses")
    ]
