module Simulation.STDP (
    STDPConf(..),
    STDPApplication(..),
    asymPotentiation,
    asymDepression
) where

import Control.Monad (liftM)
import Data.Binary


data STDPConf = STDPConf {

        stdpEnabled :: Bool,

        {- Lookup-table mapping time difference (dt) to additive weight for
         - potentiation and depression -}
        stdpPotentiation :: [Double],
        stdpDepression :: [Double],

        -- TODO: use Maybe here, to allow max weight be max in current network
        stdpMaxWeight :: Double,

        -- | We may specify a fixed frequency with which STDP should be applied
        stdpFrequency :: Maybe Int
    } deriving (Show, Eq)


{- "Standard" potentiation and depression for asymetric STDP.
 -
 - s = alpha * exp(-dt/tau), when dt < tau
 -}
asymPotentiation :: [Double]
asymPotentiation = map f [0..max_dt-1]
    where
        f dt = 1.0 * exp(-dt / max_dt)
        max_dt = 20


asymDepression :: [Double]
asymDepression = map f [0..max_dt-1]
    where
        f dt = -0.8 * exp(-dt / max_dt)
        max_dt = 20


{- We may need to configure STDP from over the network -}
instance Binary STDPConf where
    put (STDPConf en pot dep mw f) =
        put en >> put pot >> put dep >> put mw >> put f
    get = do
        en  <- get
        pot <- get
        dep <- get
        mw  <- get
        f   <- get
        return $ STDPConf en pot dep mw f



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
