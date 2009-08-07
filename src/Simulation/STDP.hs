module Simulation.STDP (
    StdpConf(..),
    StdpApplication(..),
    asymPotentiation,
    asymDepression,
    asymStdp,
    stdpWindow,
    prefireWindow,
    postfireWindow,
    potentiationMask,
    depressionMask
) where

import Control.Monad (liftM)
import Data.Binary


data StdpConf = StdpConf {

        stdpEnabled :: Bool,

        {- | STDP function sampled at integer cycles. Positive values indicate
         - potentation, while negative values indicate depression. For the two
         - parts of the function (prefire and postfire) the index corresponds
         - to dt (which starts at 0 on both sides of the firing). -}
        prefire :: [Double],
        postfire :: [Double],

        -- TODO: use Maybe here, to allow max weight be max in current network
        stdpMaxWeight :: Double,

        -- | We may specify a fixed frequency with which STDP should be applied
        stdpFrequency :: Maybe Int
    } deriving (Show, Eq)


prefireWindow :: StdpConf -> Int
prefireWindow = length . prefire

postfireWindow :: StdpConf -> Int
postfireWindow = length . postfire

stdpWindow :: StdpConf -> Int
stdpWindow stdp = prefireWindow stdp + postfireWindow stdp


{- | Mask specifying what cycles in the STDP window correspond to potentiation.
 - The first element is at the beginning of the window -}
potentiationMask :: StdpConf -> [Bool]
potentiationMask conf = map (>0.0) $ (reverse $ prefire conf) ++ (postfire conf)

{- | Mask specifying what cycles in the STDP window correspond to depression.
 - The first element is at the beginning of the window -}
depressionMask :: StdpConf -> [Bool]
depressionMask conf = map (<0.0) $ (reverse $ prefire conf) ++ (postfire conf)




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


{- "Standard" potentiation and depressing for asymmetric STDP with new
 - configuaration scheme -}
asymStdp :: StdpConf
{- TODO: may want to use Maybe here for max weight, and default to max in
 - network. The default value here is arbitrary -}
asymStdp = StdpConf True asymPotentiation asymDepression 100.0 Nothing


{- We may need to configure STDP from over the network -}
instance Binary StdpConf where
    put (StdpConf en prefire postfire mw f) =
        put en >> put prefire >> put postfire >> put mw >> put f
    get = do
        en  <- get
        prefire <- get
        postfire <- get
        mw  <- get
        f   <- get
        return $ StdpConf en prefire postfire mw f


data StdpApplication
        = StdpApply Double   -- apply with multiplier
        | StdpIgnore
    deriving (Show, Eq)


instance Binary StdpApplication where
    put (StdpApply m) = putWord8 1 >> put m
    put StdpIgnore = putWord8 2
    get = do
        tag <- getWord8
        case tag of
            1 -> liftM StdpApply get
            2 -> return StdpIgnore
            _ -> error "Incorrectly serialised StdpApplication data"
