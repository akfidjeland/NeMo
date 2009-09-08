{-# LANGUAGE ForeignFunctionInterface #-}

{- This backend uses a CUDA kernel, which is accessed through a C API. CUDA
 - requires that the computation be split up into *thread blocks*. The logical
 - network is mapped onto this computational topology by defining a number of
 - neuron partitions (ideally strongly interconnected, weakly connected
 - externally). -}

module Simulation.CUDA (initSim, deviceCount) where


import Control.Monad (when)
import Control.Monad.Writer (runWriter)
import Control.Exception (assert)
-- import Data.Either
import Data.Maybe (fromMaybe)
import Foreign.ForeignPtr (withForeignPtr)

import Construction.Network (Network)
import Construction.Izhikevich (IzhNeuron)
import Construction.Synapse (Static)
import Simulation (Simulation_Iface(..))
import Types
import qualified Util.Assocs as A (elems, keys, mapAssocs, mapElems, groupBy, densify)

import Simulation.CUDA.Address
import Simulation.CUDA.Configuration (configureKernel)
import Simulation.CUDA.DeviceProperties (deviceCount)
import qualified Simulation.CUDA.Probe as Probe (readFiring)
import Simulation.CUDA.KernelFFI as Kernel
    (stepBuffering, stepNonBuffering, applyStdp,
     printCycleCounters, elapsedMs, resetTimer, deviceDiagnostics, copyToDevice)
import Simulation.CUDA.Memory as Memory
import Simulation.CUDA.Mapping (mapNetwork)
import Simulation.CUDA.State (State(..))
import Simulation.STDP




-------------------------------------------------------------------------------


-- TODO: use same interface for all either Ptr CuRT, ForeignPtr CuRT, or just State
instance Simulation_Iface State where
    run = runCuda
    run_ = runCuda_
    step = stepCuda
    step_ = stepNonBuffering
    applyStdp sim reward = withForeignPtr (rt sim) $ \p -> Kernel.applyStdp p reward
    elapsed sim = withForeignPtr (rt sim) Kernel.elapsedMs
    resetTimer sim = withForeignPtr (rt sim) Kernel.resetTimer
    getWeights sim = Memory.getWeights sim
    diagnostics = Kernel.deviceDiagnostics . rt
    start sim = return () -- copy to device forced during initSim
    stop = terminateCuda


{- | Initialise simulation and return a function to step through the rest of it -}
initSim
    :: Maybe Int            -- ^ cluster size which mapper should be forced to use
    -> Network (IzhNeuron FT) Static
    -> TemporalResolution
    -> StdpConf
    -> IO State
initSim partitionSize net dt stdpConf = do
    -- TODO: select device?
    let usingStdp = stdpEnabled stdpConf
        ((cuNet, att), mapLog) = runWriter $ mapNetwork net usingStdp partitionSize
    -- TODO: send this upstream, so we can e.g. print to server log
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



runCuda :: State -> [[Idx]] -> IO [FiringOutput]
runCuda sim fstim = do
    mapM_ (Kernel.stepBuffering sim) fstim
    readFiring sim $! length fstim
    -- printCycleCounters sim


runCuda_ :: State -> [[Idx]] -> IO ()
runCuda_ sim fstim = do
    mapM_ (Kernel.stepNonBuffering sim) fstim
    -- printCycleCounters sim


stepCuda :: State -> [Idx] -> IO FiringOutput
stepCuda sim fstim = do
    Kernel.stepBuffering sim fstim
    [firing] <- readFiring sim 1
    return $! firing


readFiring :: State -> Time -> IO [FiringOutput]
readFiring sim ncycles = do
    (ncycles', fired) <- Probe.readFiring $ rt sim
    assert (ncycles == ncycles') $ do
    return $! densifyDeviceFiring (att sim) ncycles' fired


-- TODO: error handling: propagate errors to caller
densifyDeviceFiring :: ATT -> Int -> [(Time, DeviceIdx)] -> [FiringOutput]
densifyDeviceFiring att len fired = map FiringOutput dense
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
