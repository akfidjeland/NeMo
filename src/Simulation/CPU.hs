{-# LANGUAGE BangPatterns #-}

module Simulation.CPU (initSim) where

import GHC.Conc (numCapabilities)
import Control.Concurrent (forkIO, ThreadId)
import Control.Concurrent.MVar
import Control.Monad
import Data.Array.IO
import Data.Array.IArray
import Data.Array.Unboxed
import Data.Array.Base
import Data.List (sort)
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
    deliverThalamicInput rngacc iacc
    deliverSpikes iacc sq
    let forced = listArray bs $ densify forcedFiring idx
    fired <- update bs (forced :: UArray Idx Bool) ns iacc uacc vacc
    enqSpikes sq fired ss
    return $! FiringOutput fired


deliverThalamicInput rngacc iacc = do
    initI <- updateArray thalamicInput rngacc
    zipWithM_ (writeArray iacc) [0..] initI


deliverSpikes iacc sq = accCurrent iacc =<< deqSpikes sq


update (bmin, bmax) !fs !ns !is !us !vs = do
    let ncores = numCapabilities
    if ncores == 1
        then go bmin (bmax+1) [] fs ns is us vs
        else do
            let chunks = split ncores (bmin, bmax)
            -- TODO: factor out farming-out function
            results <- replicateM ncores newEmptyMVar
            zipWithM_ (\(bmin, bmax) result -> fork result $ go bmin bmax [] fs ns is us vs) chunks results
            -- TODO: maybe do map in reverse order, in order to make concat cheaper?
            return . concat =<< mapM takeMVar results
            where
                go !idx !idx_end acc !fs !ns !is !us !vs
                    | idx == idx_end = return $! reverse acc
                    | otherwise = do
                        -- TODO: check array indices, at least once?
                        let n = ns!idx
                        i <- unsafeRead is idx -- accumulated current
                        -- should perhaps store persistent dynamic state together, to save on lookups
                        u <- unsafeRead us idx
                        v <- unsafeRead vs idx
                        let forced = fs ! idx
                        let state = IzhState u v
                        -- TODO: change argument order
                        let (state', fired) = updateIzh forced i state n
                        unsafeWrite us idx $! stateU state'
                        unsafeWrite vs idx $! stateV state'
                        if fired
                            then go (idx+1) idx_end (idx:acc) fs ns is us vs
                            else go (idx+1) idx_end      acc  fs ns is us vs



{- Fork a thread and return result via MVar -}
fork :: MVar a -> IO a -> IO ThreadId
fork mvar action = forkIO (putMVar mvar =<< action)



{- split the workload into n chunks -}
split :: Int -> (Idx, Idx) -> [(Idx, Idx)]
split n (mn, mx) = map go [0..n-1]
    where
        len = mx + 1 - mn
        sz = len `divr` n
        go chunk = (mn + chunk*sz, mn + min ((chunk+1)*sz) len)


{- Integer division rounding up -}
divr x y = q + if r == 0 then 0 else 1
    where
        (q, r) = x `quotRem` y


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
