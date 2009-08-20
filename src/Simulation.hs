{-# LANGUAGE ExistentialQuantification #-}

{- | Common simulation interface for different backends -}

module Simulation (Simulation_Iface(..), Simulation(..)) where

import Data.Map

import Types
import Construction.Network (Network)
import Construction.Synapse (Synapse, Static)


class Simulation_Iface a where

    {- | Perform several simulation steps, without changing stimulation or STDP
     - application. The default method can be over-ridden to make use of
     - buffering. -}
    run :: a -> [[Idx]] -> IO [ProbeData]
    run sim fstim = mapM (step sim) fstim

    {- | Perform several simulation steps, but ignore outputs -}
    run_ :: a -> [[Idx]] -> IO ()
    run_ sim fstim = mapM_ (step_ sim) fstim

    step :: a -> [Idx] -> IO ProbeData

    step_ :: a -> [Idx] -> IO ()
    step_ sim fstim = step sim fstim >> return ()

    applyStdp :: a -> Double -> IO ()

    {- | Return the number of milliseconds of elapsed (wall-clock)
     - simulation time -}
    elapsed :: a -> IO Int

    resetTimer :: a -> IO ()

    getWeights :: a -> IO (Map Idx [Synapse Static])

    {- | Return a string with diagnostic data, which could be useful if the
     - backend fails for some reason -}
    diagnostics :: a -> IO String
    diagnostics _ = return "no diagnostics available"

    {- | Perform any clean-up operations -}
    -- TODO: could we make the garbage collector do this perhaps?
    terminate :: a -> IO ()
    terminate a = return ()


data Simulation = forall s . Simulation_Iface s => BS s


instance Simulation_Iface Simulation where
    run (BS s) = run s
    run_ (BS s) = run_ s
    step (BS s) = step s
    step_ (BS s) = step_ s
    applyStdp (BS s) = applyStdp s
    elapsed (BS s) = elapsed s
    resetTimer (BS s) = resetTimer s
    getWeights (BS s) = getWeights s
    diagnostics (BS s) = diagnostics s
    terminate (BS s) = terminate s
