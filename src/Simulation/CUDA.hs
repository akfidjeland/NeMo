{-# LANGUAGE TypeSynonymInstances #-}

{- This backend uses a CUDA kernel, which is accessed through a C API. CUDA
 - requires that the computation be split up into *thread blocks*. The logical
 - network is mapped onto this computational topology by defining a number of
 - neuron partitions (ideally strongly interconnected, weakly connected
 - externally). -}

module Simulation.CUDA (initSim, Kernel.deviceCount) where


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
     printCycleCounters, elapsedMs, resetTimer, freeRT, deviceCount)
import Simulation.CUDA.Memory as Memory
import Simulation.STDP (StdpConf)



-------------------------------------------------------------------------------


-- TODO: use same interface for all either Ptr CuRT, ForeignPtr CuRT, or just State
instance Simulation_Iface State where
    run = runCuda
    run_ = runCuda_
    step = stepCuda
    step_ = Kernel.stepNonBuffering
    applyStdp sim reward = Kernel.applyStdp sim reward
    elapsed = Kernel.elapsedMs
    resetTimer = Kernel.resetTimer
    getWeights sim = Memory.getWeights sim
    start sim = return () -- copy to device forced during initSim
    stop = Kernel.freeRT


{- | Initialise simulation and return a function to step through the rest of it -}
initSim
    :: Maybe Int            -- ^ cluster size which mapper should be forced to use
    -> Network IzhNeuron Static
    -> StdpConf
    -> IO State
initSim partitionSize net stdpConf = do
    let maxProbePeriod = 1000
    initMemory net partitionSize maxProbePeriod stdpConf


-------------------------------------------------------------------------------
-- Running the simulation
-------------------------------------------------------------------------------



runCuda :: State -> [[Idx]] -> IO [FiringOutput]
runCuda sim fstim = do
    mapM_ (Kernel.stepBuffering sim) fstim
    Kernel.printCycleCounters sim
    readFiring sim $! length fstim


runCuda_ :: State -> [[Idx]] -> IO ()
runCuda_ sim fstim = do
    mapM_ (Kernel.stepNonBuffering sim) fstim
    Kernel.printCycleCounters sim


stepCuda :: State -> [Idx] -> IO FiringOutput
stepCuda sim fstim = do
    Kernel.stepBuffering sim fstim
    [firing] <- readFiring sim 1
    return $! firing


readFiring :: State -> Time -> IO [FiringOutput]
readFiring sim ncycles = do
    (ncycles', fired) <- Kernel.readFiring sim
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
