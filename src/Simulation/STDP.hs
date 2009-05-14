module Simulation.STDP (
    STDPConf(..),
    STDPApplication(..)
) where

import Control.Monad (liftM)
import Data.Binary

{- Configuration for STDP, where P indicates potentiation and D indicates
 - depression. Synapse modification, s,  is compuated as follows:
 -
 - s = alpha * exp(-dt/tau), when dt < tau
 -}

data STDPConf = STDPConf {
        stdpTauP :: Int,
        stdpTauD :: Int,
        stdpAlphaP :: Double,
        stdpAlphaD :: Double,
        stdpMaxWeight :: Double,

        -- | We may specify a fixed frequency with which STDP should be applied
        stdpFrequency :: Maybe Int
    } deriving (Show, Eq)



{- We may need to conrfigure STDP from over the network -}
instance Binary STDPConf where
    put (STDPConf ap ad tp td mw f) = put ap >> put ad >> put tp
        >> put td >> put mw >> put f
    get = do
        ap <- get
        ad <- get
        tp <- get
        td <- get
        mw <- get
        f <- get
        return $ STDPConf ap ad tp td mw f



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

