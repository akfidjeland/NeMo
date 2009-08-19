module Types where

import Data.List (sort)

type FT = Double
-- type FT = Float

type Idx     = Int     -- unique indices for neurons
type Source  = Idx
type Target  = Idx
type Voltage = FT
type Current = FT
type Time    = Int     -- synchronous simulation only
type TemporalResolution = Int
type Delay   = Time

data Duration
        = Forever
        | Until Time


{- Run-time probed data -}
data ProbeData
        = FiringData [Idx]
        | FiringCount Int
        | NeuronState [(Idx, [FT])]
    deriving (Eq)

instance Show ProbeData where
    show (FiringData x)  = show $ sort x
    show (FiringCount x) = show x
    show (NeuronState x) = show x
