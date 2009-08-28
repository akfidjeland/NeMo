{-# LANGUAGE FlexibleContexts #-}

module Simulation.CPU (initSim) where

import Data.Array.IO
import Data.Array.Base

import Construction.Network
import qualified Construction.Neurons as Neurons (size, neurons, indices)
import Construction.Neuron (ndata)
import Construction.Izhikevich (IzhNeuron)
import Construction.Spiking
import Construction.Synapse
import Simulation
import Simulation.FiringStimulus
import Simulation.SpikeQueue as SQ
import Types
import qualified Util.Assocs as A (mapElems)


{- The Network data type is used when constructing the net, but is not suitable
 - for execution. When executing we need 1) fast random access 2) in-place
 - modification, hence IOArray. -}
data CpuSimulation = CpuSimulation {
        network  :: IOArray Idx (IzhNeuron FT),
        synapses :: SynapsesRT Static,
        spikes   :: SpikeQueue Static
    }



instance Simulation_Iface CpuSimulation where
    step = stepSim
    applyStdp _ _ = error "STDP not supported on CPU backend"
    -- TODO: implement these properly. The dummy definitions are needed for testing
    elapsed _ = return 0
    resetTimer _ = return ()
    getWeights _ = error "getWeights not supported on CPU backend"
    start _ = return ()



{- | Initialise simulation and return function to step through simuation -}
initSim :: Network (IzhNeuron FT) Static -> IO CpuSimulation
initSim net = mkRuntime net



{- | Perform a single simulation step. Update the state of every neuron and
 - propagate spikes -}
stepSim :: CpuSimulation -> [Idx] -> IO ProbeData
stepSim (CpuSimulation ns ss sq) forcedFiring = do
    bounds <- getBounds ns
    let idx = [fst bounds..snd bounds]
    addCurrent ns =<< deqSpikes sq
    -- TODO: combine the pre-spike delivery and the update function here
    updateArray preSpikeDelivery ns
    assoc <- getAssocs ns
    let ns' = zipWith (liftN update) (densify forcedFiring idx) assoc
    mapM_ (uncurry (unsafeWrite ns)) ns'
    assoc' <- getAssocs ns
    let fired = firingIdx assoc'
    enqSpikes sq fired ss
    return $! FiringData fired
    where
        liftN f x (y, z) = (y, f x z)
        firingIdx assoc = map fst $ filter (fired . snd) assoc



addCurrent
    :: (MArray a (n FT) m, Spiking n  FT, Ix ix)
    => a ix (n FT) -> [(ix, Current)] -> m ()
addCurrent arr current = mapM_ (aux arr) current
    where
        aux arr (idx, i) = do
            neuron <- readArray arr idx
            writeArray arr idx (addSpike i neuron)


{- | Apply function to each neuron and modify in-place -}
updateArray :: (MArray a e' m, MArray a e m, Ix i) => (e -> e) -> a i e -> m ()
updateArray f xs = getAssocs xs >>= mapM_ (modify xs f)
    where
        modify xs f (i, e) = writeArray xs i $ f e



-------------------------------------------------------------------------------
-- Runtime simulation data
-------------------------------------------------------------------------------



-- pre: neurons in ns are numbered consecutively from 0-size ns-1.
mkRuntimeN ns =
    if validIdx ns
        then newListArray (0, Neurons.size ns - 1) (map ndata (Neurons.neurons ns))
        else error "mkRuntimeN: Incorrect indices in neuron map"
    where
        validIdx ns = all (uncurry (==)) (zip [0..] (Neurons.indices ns))


mkRuntime net@(Network ns _) = do
    ns' <- mkRuntimeN ns
    let ss = mkSynapsesRT net
    sq <- mkSpikeQueue net
    return $! CpuSimulation ns' ss sq



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
