module Simulation.STDP (
    STDPConf(..),
    STDPApplication(..),
    stdpOptions
) where

import Control.Monad (liftM)
import Data.Binary
import Options

{- Configuration for STDP, where P indicates potentiation and D indicates
 - depression. Synapse modification, s,  is compuated as follows:
 -
 - s = alpha * exp(-dt/tau), when dt < tau
 -}

data STDPConf = STDPConf {

        stdpEnabled :: Bool,

        stdpTauP :: Int,
        stdpTauD :: Int,
        stdpAlphaP :: Double,
        stdpAlphaD :: Double,
        stdpMaxWeight :: Double,

        -- | We may specify a fixed frequency with which STDP should be applied
        stdpFrequency :: Maybe Int
    } deriving (Show, Eq)



{- We may need to configure STDP from over the network -}
instance Binary STDPConf where
    put (STDPConf en ap ad tp td mw f) = put en >> put ap >> put ad >> put tp
        >> put td >> put mw >> put f
    get = do
        en <- get
        ap <- get
        ad <- get
        tp <- get
        td <- get
        mw <- get
        f <- get
        return $ STDPConf en ap ad tp td mw f



data STDPApplication
        = STDPApply Double   -- apply with multiplier
        | STDPIgnore
    deriving (Show, Eq)


instance Binary STDPApplication where
    put (STDPApply m) = putWord8 1 >> put m
    put STDPIgnore = putWord8 2
    get = do
        tag <- getWord8
        case tag of
            1 -> liftM STDPApply get
            2 -> return STDPIgnore
            _ -> error "Incorrectly serialised STDPApplication data"



{- Command-line options -}

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
