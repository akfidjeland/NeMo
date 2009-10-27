{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeSynonymInstances #-}

module Simulation.CPU (initSim) where

import Control.Concurrent.MVar
import Control.Monad
import Control.Monad.ST
import Control.Parallel.Strategies
import Data.Array.IO
import Data.Array.ST
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
        let (st', firing) = stepSim st forced
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
stepSim :: CpuSimulation -> [Idx] -> (CpuSimulation, FiringOutput)
stepSim sim forcedFiring =
    let (rng', initI) = unzip $ map thalamicInput $ currentRNG sim
        (todeliver, sq1) = deqSpikes $ spikes sim
        iacc2 = accCurrent (nbounds sim) initI todeliver
        forced = densify forcedFiring [0..]
        st = state sim
        (st', fired0) = unzip $ parZipWith4 rwhnf updateIzh forced iacc2 st (network sim)
        fired = find $ zip [0..] fired0
        sq' = enqSpikes sq1 fired $ synapses sim
    in (sim { state = st', currentRNG = rng', spikes = sq' }, FiringOutput fired)
    where
        find = map fst . filter snd



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
accCurrent :: (Idx, Idx) -> [Current] -> [(Idx, Current)] -> [Current]
accCurrent bs initI current = elems $ runSTUArray accumulate
    where
        accumulate :: ST s (STUArray s Idx Current)
        accumulate = do
            iacc <- newListArray bs initI
            mapM_ (go iacc) current
            return $! iacc
        -- go arr (idx, w) = writeArray arr idx . (+w) =<< readArray arr idx
        go arr (idx, w) = do
            i <- unsafeRead arr idx
            unsafeWrite arr idx (i + w)



-------------------------------------------------------------------------------
-- Runtime simulation data
-------------------------------------------------------------------------------


{- | Initialise simulation and return function to step through simuation -}
initSim :: Network IzhNeuron Static -> IO SimState
initSim net@(Network ns _) = newIORef $ CpuSimulation ns' ss sq state rng bs
    where
        ns' = map ndata (Neurons.neurons ns)
        ss = mkSynapsesRT net
        sq = mkSpikeQueue net
        bs = (0, Neurons.size ns-1)
        us = map (initU . ndata) $ neurons net
        vs = map (initV . ndata) $ neurons net
        state = zipWith IzhState us vs
        rng = map (stateThalamic . ndata) $ neurons net




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
