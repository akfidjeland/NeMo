{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CUDA (initSim, deviceCount) where


import Control.Monad (when, forM)
import Control.Monad.Writer (runWriter)
import Control.Exception (assert)
import Data.Array.Storable (withStorableArray)
import Data.Array.MArray (newListArray)
import Data.Either
import Data.Maybe (fromMaybe)
import Foreign.ForeignPtr (withForeignPtr)

import qualified Construction.Network as N
import Construction.Izhikevich (IzhNeuron, IzhState)
import Construction.Synapse (Static)
import Simulation (Simulation_Iface(..))
import Simulation.SpikeQueue
import Types
import qualified Util.Assocs as A (elems, keys, mapAssocs, mapElems, groupBy, densify)

import Simulation.CUDA.Address
import Simulation.CUDA.Configuration (configureKernel)
import Simulation.CUDA.DeviceProperties (deviceCount)
import qualified Simulation.CUDA.Probe as Probe (readFiring, readFiringCount)
import Simulation.CUDA.KernelFFI as Kernel (stepBuffering, applyStdp, syncSimulation, printCycleCounters, elapsedMs, resetTimer, deviceDiagnostics, copyToDevice)
import Simulation.CUDA.Memory as Memory
import Simulation.CUDA.Mapping (mapNetwork)
import Simulation.CUDA.State (State(..))
import Simulation.STDP


{- This backend uses a CUDA kernel, which is accessed through a C API. CUDA
 - requires that the computation be split up into *thread blocks*. The logical
 - network is mapped onto this computational topology by defining a number of
 - neuron partitions (ideally strongly interconnected, weakly connected
 - externally). -}



-------------------------------------------------------------------------------


instance Simulation_Iface State where
    run = runCuda
    run_ = runCuda_
    step = stepCuda
    step_ = stepCuda_
    applyStdp sim reward = withForeignPtr (rt sim) $ \p -> Kernel.applyStdp p reward
    elapsed sim = withForeignPtr (rt sim) Kernel.elapsedMs
    resetTimer sim = withForeignPtr (rt sim) Kernel.resetTimer
    getWeights sim = Memory.getWeights sim
    start sim = copyToDevice (rt sim)
    stop = terminateCuda


{- | Initialise simulation and return a function to step through the rest of it -}
initSim
    :: Maybe Int            -- ^ cluster size which mapper should be forced to use
    -> N.Network (IzhNeuron FT) Static
    -> TemporalResolution
    -> StdpConf
    -> IO State
initSim partitionSize net dt stdpConf = do
    -- TODO: select device?
    let usingStdp = stdpEnabled stdpConf
        ((cuNet, att), mapLog) = runWriter $ mapNetwork net usingStdp partitionSize
    when (not $ null mapLog) $ writeFile "map.log" mapLog
    -- TODO: should we free this memory?
    configureKernel cuNet
    let maxProbePeriod = 1000
    initMemory cuNet att maxProbePeriod dt stdpConf



-- free the device, clear all memory in Sim
terminateCuda :: State -> IO ()
terminateCuda sim = withForeignPtr (rt sim) freeRT


-------------------------------------------------------------------------------
-- Running the simulation
-------------------------------------------------------------------------------



runCuda :: State -> [[Idx]] -> IO [ProbeData]
runCuda sim fstim = do
    runCuda_ sim fstim
    readFiring sim $! length fstim


runCuda_ :: State -> [[Idx]] -> IO ()
runCuda_ sim fstim = do
    sequence2_ (stepBuffering sim) fstim
    printCycleCounters sim
    where
        -- TODO: replace by sequence_
        sequence2_ m (a:as) = m a >> sequence2_ m as
        sequence2_ _ _ = return ()


stepCuda :: State -> [Idx] -> IO ProbeData
stepCuda sim fstim = do
    stepCuda_ sim fstim
    [firing] <- readFiring sim 1
    return $! firing


stepCuda_ :: State -> [Idx] -> IO ()
stepCuda_ sim fstim = stepBuffering sim fstim


readFiring :: State -> Time -> IO [ProbeData]
readFiring sim ncycles = do
    (ncycles', fired) <- Probe.readFiring $ rt sim
    assert (ncycles == ncycles') $ do
    return $! densifyDeviceFiring (att sim) ncycles' fired


-- TODO: error handling: propagate errors to caller, as in stepOne below
densifyDeviceFiring :: ATT -> Int -> [(Time, DeviceIdx)] -> [ProbeData]
densifyDeviceFiring att len fired = map FiringData dense
    where
        gidx :: [(Time, Idx)]
        gidx = A.mapElems toGlobal fired

        toGlobal :: DeviceIdx -> Idx
        toGlobal didx = fromMaybe
                (error $ "densifyDeviceFiring: neuron not found: " ++ (show didx))
                (globalIdxM att didx)

        grouped :: [(Time, [(Time, Idx)])]
        grouped = A.groupBy fst gidx

        grouped' :: [(Time, [Idx])]
        grouped' = A.mapElems (map snd) grouped

        dense :: [[Idx]]
        dense = A.densify 0 len [] grouped'
