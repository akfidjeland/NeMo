{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeSynonymInstances #-}

module Simulation.CPU (initSim) where

import Control.Monad (forM_)
import Data.Array.Storable
import Data.Array.Base
import Data.IORef

import qualified Construction.Network as Network
    (Network(Network), neurons, synapses, maxDelay)
import qualified Construction.Neurons as Neurons
    (size, neurons, indices)
import Construction.Neuron (ndata)
-- TODO: add import list
import Construction.Izhikevich
import Construction.Synapse
import Simulation
import Simulation.FiringStimulus
import Simulation.SpikeQueue as SQ
import qualified Simulation.CPU.KernelFFI as Kernel
    (RT, set, addSynapses, update, clear)
import Types
import qualified Util.Assocs as A (mapElems)


{- The Network data type is used when constructing the net, but is not suitable
 - for execution. When executing we need 1) fast random access 2) in-place
 - modification, hence IOArray. -}
data CpuSimulation = CpuSimulation {
        synapses :: SynapsesRT,
        spikes   :: SpikeQueue,
        current  :: StorableArray Int Double,
        rt       :: Kernel.RT,
        nbounds  :: (Int, Int)
    }


type SimState = IORef CpuSimulation

instance Simulation_Iface SimState where
    step sim forced = do
        st <- readIORef sim
        (st', firing) <- stepSim st forced
        writeIORef sim st'
        return $! firing
    applyStdp _ _ = error "STDP not supported on CPU backend"
    -- TODO: implement these properly. The dummy definitions are needed for testing
    elapsed _ = return 0
    resetTimer _ = return ()
    getWeights _ = error "getWeights not supported on CPU backend"
    start _ = return ()
    stop sim = Kernel.clear . rt =<< readIORef sim




{- | Perform a single simulation step. Update the state of every neuron and
 - propagate spikes -}
stepSim :: CpuSimulation -> [Idx] -> IO (CpuSimulation, FiringOutput)
stepSim sim forcedFiring = do
    let (todeliver, sq1) = deqSpikes $ spikes sim
    accCurrent (current sim) todeliver
    let fstim = densify forcedFiring [0..]
    fired <- Kernel.update (rt sim) (nbounds sim) fstim (current sim)
    let sq' = enqSpikes sq1 fired $ synapses sim
    return $! (sim { spikes = sq' }, FiringOutput fired)



{- | Accumulate current for each neuron for spikes due to be delivered right
 - now -}
accCurrent :: StorableArray Idx Current -> [(Idx, Current)] -> IO ()
accCurrent iacc spikes = do
    -- clear before accumulating
    (mn, mx) <- getBounds iacc
    mapM_ (\idx -> unsafeWrite iacc idx 0) [mn..mx]
    mapM_ (go iacc) spikes
    where
        go arr (idx, w) = do
            i <- unsafeRead arr idx
            unsafeWrite arr idx (i + w)



-------------------------------------------------------------------------------
-- Runtime simulation data
-------------------------------------------------------------------------------


{- | Initialise simulation and return function to step through simuation -}
initSim :: Network.Network IzhNeuron Static -> IO SimState
initSim net@(Network.Network ns _) = do
    rt <- Kernel.set as bs cs ds us vs sigma $ Network.maxDelay net
    setConnectivityMatrix rt $ Network.synapses net
    iacc <- newArray bounds 0
    newIORef $ CpuSimulation ss sq iacc rt bounds
    where
        ns' = map ndata (Neurons.neurons ns)
        as = map paramA ns'
        bs = map paramB ns'
        cs = map paramC ns'
        ds = map paramD ns'
        us = map initU ns'
        vs = map initV ns'
        sigma = map (maybe 0.0 id . stateSigma) ns'
        ss = mkSynapsesRT net
        sq = mkSpikeQueue net
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
