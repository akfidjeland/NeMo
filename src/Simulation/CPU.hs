{-# LANGUAGE TypeSynonymInstances #-}

module Simulation.CPU (initSim) where

import Control.Monad (forM_)

import qualified Construction.Network as Network
    (Network(Network), synapses, maxDelay)
import qualified Construction.Neurons as Neurons
    (size, neurons)
import Construction.Neuron (ndata)
import Construction.Izhikevich
import Construction.Synapse (Static)
import Simulation
import qualified Simulation.CPU.KernelFFI as Kernel
import Types


data CpuSimulation = CpuSimulation {
        rt       :: Kernel.RT,
        fstim    :: Kernel.StimulusBuffer
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
    return . FiringOutput =<< Kernel.step (rt sim) (fstim sim) forcedFiring



-------------------------------------------------------------------------------
-- Runtime simulation data
-------------------------------------------------------------------------------


initSim :: Network.Network IzhNeuron Static -> IO CpuSimulation
initSim net@(Network.Network ns _) = do
    rt <- Kernel.set as bs cs ds us vs sigma $ Network.maxDelay net
    setConnectivityMatrix rt $ Network.synapses net
    fstim <- Kernel.newStimulusBuffer $ Neurons.size ns
    return $ CpuSimulation rt fstim
    where
        ns' = map ndata (Neurons.neurons ns)
        as = map paramA ns'
        bs = map paramB ns'
        cs = map paramC ns'
        ds = map paramD ns'
        us = map initU ns'
        vs = map initV ns'
        sigma = map (maybe 0.0 id . stateSigma) ns'



setConnectivityMatrix rt ss0 =
    forM_ ss0 $ \(src, ss1) ->
        forM_ ss1 $ \(delay, ss2) -> do
            let (targets, weights) = unzip $ map strip ss2
            Kernel.addSynapses rt src delay targets weights
    where
        strip (t, w, _, _) = (t, w)
