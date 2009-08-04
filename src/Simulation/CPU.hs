{-# LANGUAGE FlexibleContexts #-}

module Simulation.CPU (initSim) where

import Data.Array.IO
import Data.Array.Base
import qualified Data.Map as Map
import qualified Data.Set as Set
import Control.Monad

import Construction.Network
import qualified Construction.Neurons as Neurons (size, neurons, indices)
import Construction.Neuron (NeuronProbe, mergeProbeFs, ndata)
import Construction.Spiking
import Construction.Synapse
import Simulation.Common
import Simulation.FiringStimulus
import Simulation.SpikeQueue as SQ
import Types
import qualified Util.Assocs as A (mapElems)



{- | Initialise simulation and return function to step through simuation -}
initSim
    :: (Spiking n FT, Conductive s, NeuronProbe a n FT)
    => Network (n FT) s
    -> Probe
    -> ProbeFn a
    -> IO Simulation
initSim net pidx pfn = do
    netRT <- mkRuntime net
    return $ Simulation 1
        -- TODO: add support for STDP
        (\fstim _ -> mapM (step netRT) fstim)
        -- TODO: add proper timing
        (return 0)
        (return ())
        (error "getWeights not implemented in 'CPU' backend")
        (error "diagnostics not implemented in 'CPU' backend")
        (return ())
    where
        step netRT f = stepSim netRT pset f >>= return . (outputFn pfn)

        pset = getProbeSet pidx net

        outputFn Firing ns     = FiringData $ map fst $ filter (fired . snd) ns
        outputFn (State ps) ns = NeuronState $ A.mapElems (mergeProbeFs ps) ns



{- | Perform a single simulation step. Update the state of every neuron and
 - propagate spikes -}
stepSim :: (Spiking n FT, Conductive s)
    => NetworkRT (n FT) s
    -> ProbeSet
    -> [Idx]               -- ^ firing stimulus
    -> IO [(Idx, (n FT))]
stepSim (NetworkRT ns ss sq) probe forcedFiring = do
    bounds <- getBounds ns
    let idx = [fst bounds..snd bounds]
    addCurrent ns =<< deqSpikes sq
    -- TODO: combine the pre-spike delivery and the update function here
    updateArray preSpikeDelivery ns
    assoc <- getAssocs ns
    let ns' = zipWith (liftN update) (densify forcedFiring idx) assoc
    mapM_ (uncurry (unsafeWrite ns)) ns'
    assoc' <- getAssocs ns
    enqSpikes sq (firingIdx assoc') ss
    return $ filter (((flip Set.member) probe) . fst) assoc'
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


{- The Network data type is used when constructing the net, but is not suitable
 - for execution. When executing we need 1) fast random access 2) in-place
 - modification, hence IOArray. -}
data NetworkRT n s = NetworkRT {
        network  :: IOArray Idx n,
        synapses :: SynapsesRT s,
        spikes   :: SpikeQueue s
    }


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
    return $ NetworkRT ns' ss sq



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
