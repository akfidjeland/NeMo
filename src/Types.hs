module Types where

import Data.List (sort)

type FT = Double
-- type FT = Float

type Idx     = Int     -- unique indices for neurons
type Voltage = FT
type Current = FT
type Time    = Int     -- synchronous simulation only
type TemporalResolution = Int
type Delay   = Time

data Duration
        = Forever
        | Until Time

data Probe
        = All
        | Only [Idx]


{- User specfication of desired data -}
-- TODO: is this really needed? Could just derive the probed data from the return type
data ProbeFn a
        = Firing           -- ^ Return firing status
        | ProbeFiringCount -- ^ Return only the number of neurons that fired
        | State [a]        -- ^ Return dynamic state of neuron
    deriving (Show)

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
