{-# LANGUAGE BangPatterns #-}

module Simulation.CPU (initSim) where

import Control.Monad (zipWithM_)
import Data.Array.IO
import Data.Array.IArray
import Data.Array.Base
import System.Random (StdGen)

import Construction.Network
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
        network  :: Array Idx IzhNeuron,
        synapses :: SynapsesRT,
        spikes   :: SpikeQueue,
        currentAcc :: IOUArray Idx FT,
        currentU :: IOUArray Idx FT,
        currentV :: IOUArray Idx FT,
        -- TODO: also wrap whole array in maybe so we can bypass one pass over array
        currentRNG :: IOArray Idx (Maybe (Thalamic FT))
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
initSim :: Network IzhNeuron Static -> IO CpuSimulation
initSim net = mkRuntime net



{- | Perform a single simulation step. Update the state of every neuron and
 - propagate spikes -}
stepSim :: CpuSimulation -> [Idx] -> IO FiringOutput
stepSim (CpuSimulation ns ss sq iacc uacc vacc rngacc) forcedFiring = do
    let bs = bounds ns
    let idx = [fst bs..snd bs]
    initI <- updateArray thalamicInput rngacc
    zipWithM_ (writeArray iacc) [0..] initI
    accCurrent iacc =<< deqSpikes sq
    -- TODO: avoid need to densify
    let forced = densify forcedFiring idx
    fired <- update forced ns iacc uacc vacc
    enqSpikes sq fired ss
    return $! FiringOutput fired


update fs !ns !is !us !vs = go fs 0 ns is us vs
    where
        go [] _ _ _ _ _ = return []
        go (forced:fs) !idx !ns !is !us !vs = do
            -- TODO: check array indices, at least once?
            let n = ns!idx
            i <- unsafeRead is idx -- accumulated current
            -- should perhaps store persistent dynamic state together, to save on lookups
            u <- unsafeRead us idx
            v <- unsafeRead vs idx
            let state = IzhState u v
            -- TODO: change argument order
            let (state', fired) = updateIzh forced i state n
            unsafeWrite us idx $! stateU state'
            unsafeWrite vs idx $! stateV state'
            fs <- go fs (idx+1) ns is us vs
            if fired
                then return $! idx : fs
                else return $!       fs


{- | Accumulate current for each neuron for spikes due to be delivered right
 - now -}
accCurrent :: IOUArray Idx Current -> [(Idx, Current)] -> IO ()
accCurrent arr current = mapM_ go current
    where
        -- go arr (idx, w) = writeArray arr idx . (+w) =<< readArray arr idx
        go (idx, w) = do
            i <- readArray arr idx
            writeArray arr idx (i + w)



{- | Apply function to each neuron and modify in-place -}
updateArray :: (MArray a e' m, MArray a e m, Ix i) => (e -> (e, b)) -> a i e -> m [b]
updateArray f xs = getAssocs xs >>= mapM (modify xs f)
    where
        modify xs f (i, e) = do
            let (e', val) = f e
            writeArray xs i e'
            return $! val



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


mkRuntime net@(Network ns _) = do
    let ns' = mkRuntimeN ns
    let ss = mkSynapsesRT net
    sq <- mkSpikeQueue net
    -- TODO: do the same bounds checking as for mkRuntimeN
    iacc <- newListArray (0, Neurons.size ns-1) (repeat 0)
    uacc <- newListArray (0, Neurons.size ns-1) $ map (initU . ndata) $ neurons net
    vacc <- newListArray (0, Neurons.size ns-1) $ map (initV . ndata) $ neurons net
    rngacc <- newListArray (0, Neurons.size ns-1) $ map (stateThalamic . ndata) $ neurons net
    return $! CpuSimulation ns' ss sq iacc uacc vacc rngacc




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
