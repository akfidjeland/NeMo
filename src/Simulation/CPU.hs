{-# LANGUAGE TypeSynonymInstances #-}

module Simulation.CPU (initSim) where

import Control.Monad (forM_)

import qualified Construction.Network as Network
    (Network(Network), synapses)
import qualified Construction.Neurons as Neurons
    (size, neurons)
import Construction.Neuron (ndata)
import Construction.Izhikevich
import Construction.Synapse (Static)
import Data.List (unzip4)
import Simulation
import Simulation.STDP (StdpConf)
import qualified Simulation.CPU.KernelFFI as Kernel
import Types


-- TODO: move this into Kernel
data CpuSimulation = CpuSimulation {
        rt       :: Kernel.RT,
        fstim    :: Kernel.StimulusBuffer
    }



instance Simulation_Iface CpuSimulation where
    step = stepSim
    step_ = stepSim_
    applyStdp _ _ = error "STDP not supported on CPU backend"
    elapsed sim = Kernel.elapsedMs $ rt sim
    resetTimer sim = Kernel.resetTimer $ rt sim
    getWeights _ = error "getWeights not supported on CPU backend"
    start = Kernel.start . rt
    stop = Kernel.clear . rt




{- | Perform a single simulation step. Update the state of every neuron and
 - propagate spikes. Do not return firing. -}
stepSim_ :: CpuSimulation -> [Idx] -> IO ()
stepSim_ sim forcedFiring = Kernel.step (rt sim) (fstim sim) forcedFiring


{- | - Perform a single simulation step. Update the state of every neuron and
 - propagate spikes. Return firing. -}
stepSim :: CpuSimulation -> [Idx] -> IO FiringOutput
stepSim sim fstim = do
    stepSim_ sim fstim
    return . FiringOutput =<< Kernel.readFiring (rt sim)



-------------------------------------------------------------------------------
-- Runtime simulation data
-------------------------------------------------------------------------------


initSim
    :: Network.Network IzhNeuron Static
    -> StdpConf
    -> IO CpuSimulation
initSim net@(Network.Network ns _) stdp = do
    rt <- Kernel.setNetwork as bs cs ds us vs sigma
    setConnectivityMatrix rt $ Network.synapses net
    fstim <- Kernel.newStimulusBuffer $ ncount
    Kernel.configureStdp rt stdp
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
        ncount = Neurons.size ns



setConnectivityMatrix rt ss0 =
    forM_ ss0 $ \(src, ss1) ->
        forM_ ss1 $ \(delay, ss2) -> do
            let (targets, weights, plastic, _) = unzip4 ss2
            Kernel.addSynapses rt src delay targets weights plastic
