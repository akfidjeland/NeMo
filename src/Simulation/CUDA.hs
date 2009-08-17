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
import Construction.Topology (Topology(NoTopology))
import Simulation (Simulation_Iface(..))
import Simulation.SpikeQueue
import Types
import qualified Util.Assocs as A (elems, keys, mapAssocs, mapElems, groupBy, densify)

import Simulation.CUDA.Address
import Simulation.CUDA.Configuration (configureKernel)
import Simulation.CUDA.DeviceProperties (deviceCount)
import qualified Simulation.CUDA.Probe as Probe (readFiring, readFiringCount)
import Simulation.CUDA.KernelFFI as Kernel (c_step, applyStdp, syncSimulation, printCycleCounters, elapsedMs, resetTimer, deviceDiagnostics)
import Simulation.CUDA.Memory as Memory
import Simulation.CUDA.Mapping (mapNetwork)
import Simulation.STDP


{- This backend uses a CUDA kernel, which is accessed through a C API. CUDA
 - requires that the computation be split up into *thread blocks*. The logical
 - network is mapped onto this computational topology by defining a number of
 - neuron partitions (ideally strongly interconnected, weakly connected
 - externally). -}



-------------------------------------------------------------------------------


-- TODO: rename
instance Simulation_Iface SimData where
    run = runCuda
    step = stepCuda
    applyStdp sim reward = withForeignPtr (rt sim) $ \p -> Kernel.applyStdp p reward
    elapsed sim = withForeignPtr (rt sim) Kernel.elapsedMs
    resetTimer sim = withForeignPtr (rt sim) Kernel.resetTimer
    getWeights sim = do
        ns <- Memory.getWeights sim
        return $! N.Network ns NoTopology


{- | Initialise simulation and return a function to step through the rest of it -}
initSim
    :: Maybe Int            -- ^ cluster size which mapper should be forced to use
    -> N.Network (IzhNeuron FT) Static
    -> TemporalResolution
    -> StdpConf
    -> IO SimData
initSim partitionSize net dt stdpConf = do
    -- TODO: select device?
    let usingStdp = stdpEnabled stdpConf
        ((cuNet, att), mapLog) = runWriter $ mapNetwork net usingStdp partitionSize
    when (not $ null mapLog) $ writeFile "map.log" mapLog
    -- TODO: should we free this memory?
    configureKernel cuNet
    let maxProbePeriod = 1000
    initMemory cuNet att maxProbePeriod dt stdpConf



-------------------------------------------------------------------------------
-- Running the simulation
-------------------------------------------------------------------------------



runCuda :: SimData -> [[Idx]] -> IO [ProbeData]
runCuda sim fstim = do
    sequence2_ (stepBuffering sim) fstim [0..]
    withForeignPtr (rt sim) $ \rtPtr -> do
    printCycleCounters rtPtr
    readFiring sim $ length fstim
    where
        steps = length fstim

        sequence2_ m (a:as) (b:bs) = m a b >> sequence2_ m as bs
        sequence2_ _ _  _ = return ()


stepCuda :: SimData -> [Idx] -> IO ProbeData
stepCuda sim fstim = do
    stepBuffering sim fstim 0
    [firing] <- readFiring sim 1
    return $! firing


readFiring :: SimData -> Time -> IO [ProbeData]
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



{- Run possibly failing computation, and propagate any errors with additional
 - prefix -}
rethrow :: (Monad m) => String -> (a -> Either String b) -> a -> m b
rethrow prefix f x = either (fail . (++) (prefix ++ ": ")) return (f x)



-- TODO: move into KernelFFI
{- | Perform a single simulation step, while buffering firing data on the
 - device, rather than reading it back to the host -}
stepBuffering sim fstim currentCycle = do
    let flen = length fstim
        fbounds = (0, flen-1)
    fstim <- forM fstim $ rethrow "firing stimulus" $ deviceIdxM (att sim)
    fsPIdxArr <- newListArray fbounds (map (fromIntegral . partitionIdx) fstim)
    fsNIdxArr <- newListArray fbounds (map (fromIntegral . neuronIdx) fstim)
    withStorableArray fsPIdxArr  $ \fsPIdxPtr -> do
    withStorableArray fsNIdxArr  $ \fsNIdxPtr -> do
    withForeignPtr (rt sim)      $ \rtPtr     -> do
    kernelStatus <- c_step rtPtr currentCycle
        (dt sim) (fromIntegral flen) fsPIdxPtr fsNIdxPtr
    when (kernelStatus /= 0) $ fail "Backend error"
