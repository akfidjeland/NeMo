module Simulation.STDP (
    StdpConf(..),
    asymPotentiation,
    asymDepression,
    asymStdp,
    stdpWindow,
    prefireWindow,
    postfireWindow,
    Reward
) where

import Construction.Synapse (Synapse, Static, excitatory)

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
        stdpMinWeight :: Double,

        -- | We may specify a fixed frequency with which STDP should be applied
        stdpFrequency :: Maybe Int
    } deriving (Show, Eq)


type Reward = Double


prefireWindow :: StdpConf -> Int
prefireWindow = length . prefire

postfireWindow :: StdpConf -> Int
postfireWindow = length . postfire

stdpWindow :: StdpConf -> Int
stdpWindow stdp = prefireWindow stdp + postfireWindow stdp


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
asymStdp = StdpConf True asymPotentiation asymDepression 100.0 (-100.0) Nothing
