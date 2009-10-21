{-# LANGUAGE FlexibleContexts #-}

module Simulation.CPU (initSim) where

import Control.Monad (zipWithM_)
import Data.Array.IO
import Data.Array.Base

import Construction.Network
import qualified Construction.Neurons as Neurons (size, neurons, indices)
import Construction.Neuron (ndata)
-- TODO: add import list
import Construction.Izhikevich
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
        synapses :: SynapsesRT,
        spikes   :: SpikeQueue,
        currentAcc :: IOUArray Idx FT
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
stepSim :: CpuSimulation -> [Idx] -> IO FiringOutput
stepSim (CpuSimulation ns ss sq iacc) forcedFiring = do
    bounds <- getBounds ns
    let idx = [fst bounds..snd bounds]
    -- TODO: accumulate current into separate accumulator
    -- TODO: make use of that value in the state update instead of internal
    -- TODO: clear current accumulation
    {-
    -}
    updateArray izhPreSpikeDelivery ns
    nselem <- getElems ns
    let initI = map stateI nselem
    zipWithM_ (writeArray iacc) [0..] initI
    -- clearArray iacc
    -- TODO: set thalamic input here
    accCurrent iacc =<< deqSpikes sq
    ivals <- getElems iacc -- list of current for each neuron
    -- addCurrent ns =<< deqSpikes sq
    -- TODO: combine the pre-spike delivery and the update function here
    assoc <- getAssocs ns
    -- let ivals = repeat 0
    let ns' = zipWith3 (liftN updateIzh) (densify forcedFiring idx) ivals assoc
    mapM_ (uncurry (unsafeWrite ns)) ns'
    assoc' <- getAssocs ns
    let fired = firingIdx assoc'
    enqSpikes sq fired ss
    return $! FiringOutput fired
    where
        liftN f x i (y, z) = (y, f x i z)
        firingIdx assoc = map fst $ filter (stateF . snd) assoc

{-
clearArray = setArray 0

setArray val arr = do
    (mn,mx) <- getBounds arr
    mapM_ (\i -> writeArray arr i val) [mn..mx]
-}

{- | Accumulate current for each neuron for spikes due to be delivered right
 - now -}
accCurrent :: IOUArray Idx Current -> [(Idx, Current)] -> IO ()
accCurrent arr current = mapM_ go current
    where
        -- go arr (idx, w) = writeArray arr idx . (+w) =<< readArray arr idx
        go (idx, w) = do
            i <- readArray arr idx
            writeArray arr idx (i + w)


-- addCurrent
--    :: (MArray a (n FT) m, Spiking n  FT, Ix ix)
--    => a ix (n FT) -> [(ix, Current)] -> m ()
addCurrent arr current = mapM_ (aux arr) current
    where
        aux arr (idx, i) = do
            neuron <- readArray arr idx
            writeArray arr idx (addSpikeIzh i neuron)


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
    -- TODO: do the same bounds checking as for mkRuntimeN
    iacc <- newListArray (0, Neurons.size ns-1) (repeat 0)
    return $! CpuSimulation ns' ss sq iacc




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
