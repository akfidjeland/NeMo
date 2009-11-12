{-# LANGUAGE TypeSynonymInstances #-}

module Simulation.CPU (initSim) where

import Control.Monad (forM_)

import qualified Construction.Network as Network
    (Network(Network), neurons, synapses, maxDelay)
import qualified Construction.Neurons as Neurons
    (size, neurons, indices)
import Construction.Neuron (ndata)
import Construction.Izhikevich
import Construction.Synapse (Static)
import Simulation
import qualified Simulation.CPU.KernelFFI as Kernel
    (RT, set, addSynapses, step, clear)
import Types


{- The Network data type is used when constructing the net, but is not suitable
 - for execution. When executing we need 1) fast random access 2) in-place
 - modification, hence IOArray. -}
data CpuSimulation = CpuSimulation {
        rt       :: Kernel.RT,
        nbounds  :: (Int, Int)
    }



instance Simulation_Iface CpuSimulation where
    step = stepSim
    applyStdp _ _ = error "STDP not supported on CPU backend"
    -- TODO: implement these properly. The dummy definitions are needed for testing
    elapsed _ = return 0
    resetTimer _ = return ()
    getWeights _ = error "getWeights not supported on CPU backend"
    start _ = return ()
    stop = Kernel.clear . rt




{- | Perform a single simulation step. Update the state of every neuron and
 - propagate spikes -}
stepSim :: CpuSimulation -> [Idx] -> IO FiringOutput
stepSim sim forcedFiring =
    let fstim = densify forcedFiring [0..] in
    return . FiringOutput =<< Kernel.step (rt sim) (nbounds sim) fstim



-------------------------------------------------------------------------------
-- Runtime simulation data
-------------------------------------------------------------------------------


{- | Initialise simulation and return function to step through simuation -}
initSim :: Network.Network IzhNeuron Static -> IO CpuSimulation
initSim net@(Network.Network ns _) = do
    rt <- Kernel.set as bs cs ds us vs sigma $ Network.maxDelay net
    setConnectivityMatrix rt $ Network.synapses net
    return $ CpuSimulation rt bounds
    where
        ns' = map ndata (Neurons.neurons ns)
        as = map paramA ns'
        bs = map paramB ns'
        cs = map paramC ns'
        ds = map paramD ns'
        us = map initU ns'
        vs = map initV ns'
        sigma = map (maybe 0.0 id . stateSigma) ns'
        bounds = (0, Neurons.size ns-1)



setConnectivityMatrix rt ss0 =
    forM_ ss0 $ \(src, ss1) ->
        forM_ ss1 $ \(delay, ss2) -> do
            let (targets, weights) = unzip $ map strip ss2
            Kernel.addSynapses rt src delay targets weights
    where
        strip (t, w, _, _) = (t, w)


-------------------------------------------------------------------------------
-- Simulation utility functions
-------------------------------------------------------------------------------


{- pre: sorted xs
        sorted ys
        xs `subset` ys -}
densify :: (Ord ix) => [ix] -> [ix] -> [Bool]
densify [] ys = map (\_ -> False) ys
densify xs [] = error "densify: sparse list contains out-of-bounds indices"
densify (x:xs) (y:ys)
        | x > y     = False : (densify (x:xs) ys)
        | x == y    = True : (densify xs ys)
        | otherwise = error "densify: sparse list does not match dense list"
