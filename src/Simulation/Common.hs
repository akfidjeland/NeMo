{-# LANGUAGE CPP #-}

module Simulation.Common where

import qualified Data.Set as Set

import Construction.Network
import Simulation.STDP (STDPConf, STDPApplication)
import Types

-- TODO: move to Simulation.Run
data Backend
        = CPU                   -- ^ (multi-core) CPU
#if defined(CUDA_ENABLED)
        | CUDA                  -- ^ CUDA-enabled GPU
#endif
        | RemoteHost String Int -- ^ some other machine on specified port
    deriving (Eq)



type ProbeSet = Set.Set Idx

-- | Return set of neurons to probe
getProbeSet :: Probe -> Network n s -> ProbeSet
getProbeSet All net = Set.fromList [0..size net-1]
getProbeSet (Only xs) _ = Set.fromList xs


-- | Return lazy list of cycles
cycles :: Duration -> [Time]
cycles Forever = [0..]
cycles (Until end)
    | end < 0   = error "cycles: negative duration"
    | otherwise = [0..end-1]


-- | Apply function for a given duration
sample :: Duration -> [a] -> [a]
sample Forever xs = xs
sample (Until end) xs = take end xs



{- Each backend may have a different natural chunk of data it processes, the
 - step size. For example if the backend is conneted to some machine over a
 - network it might reasonable have a large step size to avoid being swamped by
 - communication overheads. -}
data Simulation = Simulation {
        stepSize :: Int,
        runStep  :: [[Idx]] -> [STDPApplication] -> IO [ProbeData],

        -- | Return the number of milliseconds of elapsed (wall-clock)
        -- simulation time
        elapsed :: IO Int,

        resetTimer :: IO (),

        -- | Perform any clean-up operations
        -- TODO: could we make the garbage collector do this perhaps?
        closeSim :: IO ()
    }


type SimulationInit n s
    = Network n s
    -> TemporalResolution   -- sub-step resolution
    -> Maybe STDPConf
    -> IO Simulation

-- TODO: rename, this is really multiple steps handled in one go.
type SimulationStep
    = [[Idx]]               -- ^ firing stimulus
    -> [STDPApplication]
    -> IO [ProbeData]
