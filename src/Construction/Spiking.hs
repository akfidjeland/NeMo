{-# LANGUAGE MultiParamTypeClasses #-}

module Construction.Spiking where

class Spiking n f where
    -- TODO: fired should be a run-time structure only.
    fired :: n f -> Bool
    update :: Bool -> n f -> n f    -- ^ 1st arg for forcing firing
    addSpike :: f -> n f -> n f

    {- | neuron update hook, to update state before spike delivery -}
    -- TODO: run this inside simulation monad (which can draw on random numbers)
    preSpikeDelivery :: n f -> n f
