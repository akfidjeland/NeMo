{-# LANGUAGE TypeSynonymInstances #-}

{- This backend uses a CUDA kernel, which is accessed through a C API. CUDA
 - requires that the computation be split up into *thread blocks*. The logical
 - network is mapped onto this computational topology by defining a number of
 - neuron partitions (ideally strongly interconnected, weakly connected
 - externally). -}

module Simulation.CUDA (initSim) where


import Control.Exception (assert)

import Construction.Network (Network)
import Construction.Izhikevich (IzhNeuron)
import Construction.Synapse (Static)
import Simulation (Simulation_Iface(..))
import Types
import qualified Util.Assocs as A (elems, keys, mapAssocs, mapElems, groupBy, densify)

import Simulation.CUDA.Address
import qualified Simulation.CUDA.KernelFFI as Kernel
    (stepBuffering, stepNonBuffering, applyStdp, readFiring,
     elapsedMs, resetTimer, freeRT)
import Simulation.CUDA.Memory as Memory (initSim, getWeights, State(rtdata))
import Simulation.STDP (StdpConf)



-------------------------------------------------------------------------------


-- TODO: use same interface for all either Ptr CuRT, ForeignPtr CuRT, or just State
instance Simulation_Iface State where
    run = runCuda
    run_ = runCuda_
    step = stepCuda
    step_ sim fstim = Kernel.stepNonBuffering (rtdata sim) fstim
    applyStdp sim reward = Kernel.applyStdp (rtdata sim) reward
    elapsed = Kernel.elapsedMs . rtdata
    resetTimer = Kernel.resetTimer . rtdata
    getWeights = Memory.getWeights
    start sim = return () -- copy to device forced during initSim
    stop = Kernel.freeRT . rtdata


-------------------------------------------------------------------------------
-- Running the simulation
-------------------------------------------------------------------------------



runCuda :: State -> [[Idx]] -> IO [FiringOutput]
runCuda sim fstim = do
    mapM_ (Kernel.stepBuffering $ rtdata sim) fstim
    readFiring sim $! length fstim


runCuda_ :: State -> [[Idx]] -> IO ()
runCuda_ sim fstim = do
    mapM_ (Kernel.stepNonBuffering $ rtdata sim) fstim


stepCuda :: State -> [Idx] -> IO FiringOutput
stepCuda sim fstim = do
    Kernel.stepBuffering (rtdata sim) fstim
    [firing] <- readFiring sim 1
    return $! firing


readFiring :: State -> Time -> IO [FiringOutput]
readFiring sim ncycles = do
    (ncycles', fired) <- Kernel.readFiring $ rtdata sim
    assert (ncycles == ncycles') $ do
    return $! densifyDeviceFiring ncycles' fired


-- TODO: error handling: propagate errors to caller
densifyDeviceFiring :: Int -> [(Time, Idx)] -> [FiringOutput]
densifyDeviceFiring len fired = map FiringOutput dense
    where
        grouped :: [(Time, [(Time, Idx)])]
        grouped = A.groupBy fst fired

        grouped' :: [(Time, [Idx])]
        grouped' = A.mapElems (map snd) grouped

        dense :: [[Idx]]
        dense = A.densify 0 len [] grouped'
