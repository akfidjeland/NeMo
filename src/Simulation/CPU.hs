{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeSynonymInstances #-}

module Simulation.CPU (initSim) where

import Control.Concurrent.MVar
import Control.Monad
import Control.Monad.ST
import Control.Parallel.Strategies
import Data.Array.IO
import Data.Array.IArray
import Data.Array.Unboxed
import Data.Array.Base
import Data.List (sort, zipWith4, zipWith5)
import Data.IORef
import System.Random (StdGen)

import Construction.Network (Network(Network), neurons)
import qualified Construction.Neurons as Neurons (size, neurons, indices)
import Construction.Neuron (ndata)
-- TODO: add import list
import Construction.Izhikevich
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
        network  :: [IzhNeuron],
        synapses :: SynapsesRT,
        spikes   :: SpikeQueue,
        state    :: [IzhState],
        -- TODO: also wrap whole array in maybe so we can bypass one pass over array
        currentRNG :: [Maybe (Thalamic FT)],
        nbounds :: (Int, Int)
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





{- | Perform a single simulation step. Update the state of every neuron and
 - propagate spikes -}
stepSim :: CpuSimulation -> [Idx] -> IO (CpuSimulation, FiringOutput)
stepSim sim forcedFiring = do
    let ns = network sim
    (rng', iacc) <- deliverThalamicInput (nbounds sim) (currentRNG sim)
    let sq = spikes sim
    deliverSpikes iacc sq
    iacc2 <- getElems iacc
    let forced = densify forcedFiring [0..]
        st = state sim
        (st', fired0) = unzip $ parZipWith4 rwhnf updateIzh forced iacc2 st ns
        fired = find $ zip [0..] fired0
    enqSpikes sq fired $ synapses sim
    return $! (sim { state = st', currentRNG = rng' }, FiringOutput fired)
    where
        find = map fst . filter snd



-- deliverThalamicInput rng = unzip $ map thalamicInput rng
deliverThalamicInput bs rng = do
    let (rng', initI) = unzip $ map thalamicInput rng
    iacc <- newListArray bs initI
    return $! (rng', iacc)



deliverSpikes iacc sq = accCurrent iacc =<< deqSpikes sq



{- | Zips together two lists using a function, and evaluates the result list in
 - parallel. -}
parZipWith4 strat z as bs cs ds =
    zipWith4 z as bs cs ds `using` parList strat

{- | Zips together two lists using a function, and evaluates the result list in
 - parallel chunks. -}
parChunkZipWith4 n strat z as bs cs ds =
    zipWith4 z as bs cs ds `using` parListChunk n strat



{- | Accumulate current for each neuron for spikes due to be delivered right
 - now -}
accCurrent :: IOUArray Idx Current -> [(Idx, Current)] -> IO ()
accCurrent arr current = mapM_ go current
    where
        -- go arr (idx, w) = writeArray arr idx . (+w) =<< readArray arr idx
        go (idx, w) = do
            i <- readArray arr idx
            writeArray arr idx (i + w)



-------------------------------------------------------------------------------
-- Runtime simulation data
-------------------------------------------------------------------------------



-- pre: neurons in ns are numbered consecutively from 0-size ns-1.
mkRuntimeN ns =
    if validIdx ns
        then listArray (0, Neurons.size ns - 1) (map ndata (Neurons.neurons ns))
        else error "mkRuntimeN: Incorrect indices in neuron map"
    where
        validIdx ns = all (uncurry (==)) (zip [0..] (Neurons.indices ns))


-- TODO: move out of monad
{- | Initialise simulation and return function to step through simuation -}
initSim :: Network IzhNeuron Static -> IO SimState
initSim net@(Network ns _) = do
    let ns' = map ndata (Neurons.neurons ns)
    let ss = mkSynapsesRT net
    sq <- mkSpikeQueue net
    -- TODO: do the same bounds checking as for mkRuntimeN
    let bs = (0, Neurons.size ns-1)
        us = map (initU . ndata) $ neurons net
        vs = map (initV . ndata) $ neurons net
        state = zipWith IzhState us vs
        rng = map (stateThalamic . ndata) $ neurons net

    newIORef $ CpuSimulation ns' ss sq state rng bs




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
