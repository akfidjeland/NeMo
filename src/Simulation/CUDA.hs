{-# LANGUAGE ForeignFunctionInterface #-}

module Simulation.CUDA (
    initSim,
    deviceCount
) where

import System.IO

import qualified Control.Exception as E (catch)
import Control.Monad
import Control.Monad.Error
import Control.Monad.Writer (runWriter)
import Data.Array.Storable (withStorableArray)
import Data.Array.MArray (newArray, newListArray)
import Data.Either
import Data.List (insert, groupBy, zipWith4, foldl')
import Data.Maybe (fromMaybe, isJust)
import Foreign.C.Types
import Foreign.ForeignPtr (withForeignPtr)

import qualified Construction.Network as N
import Construction.Izhikevich (IzhNeuron, IzhState)
import Construction.Synapse (Static)
import Construction.Topology (Topology(..))
import Simulation.Common (Simulation(..))
import Simulation.SpikeQueue
import Types
import qualified Util.Assocs as A (elems, keys, mapAssocs, mapElems, groupBy, densify)

import Simulation.CUDA.Address
import Simulation.CUDA.Configuration (configureKernel)
import Simulation.CUDA.DeviceProperties (deviceCount)
import Simulation.CUDA.Probe (readFiring, readFiringCount)
import Simulation.CUDA.KernelFFI as Kernel (c_step, syncSimulation, printCycleCounters, elapsedMs, resetTimer, deviceDiagnostics)
import Simulation.CUDA.Memory as Memory
import Simulation.CUDA.Mapping (mapNetwork)
import Simulation.STDP



{- This backend uses a CUDA kernel, which is accessed through a C API. CUDA
 - requires that the computation be split up into *thread blocks*. The logical
 - network is mapped onto this computational topology by defining a number of
 - neuron clusters (ideally strongly interconnected, weakly connected
 - externally).
 -
 - Some notation:
 -
 - prefix c : cluster
 - prefix d/h : device or host (memory)
 - prefix d/s : dynamic or static
 -}



-------------------------------------------------------------------------------


{- | Initialise simulation and return a function to step through the rest of it -}
initSim
    :: N.Network (IzhNeuron FT) Static
    -> Probe
    -> Maybe (ProbeFn IzhState) -- ^ read back firing data of some type
    -> TemporalResolution
    -> Bool
    -> Maybe Int            -- ^ cluster size which mapper should be forced to use
    -> STDPConf
    -> IO Simulation
initSim net probeIdx probeF dt verbose partitionSize stdpConf = do
    -- TODO: select device?
    let usingSTDP = stdpEnabled stdpConf
        ((cuNet, att), mapLog) = runWriter $ mapNetwork net usingSTDP partitionSize
    when (not $ null mapLog) $ writeFile "map.log" mapLog
    -- TODO: should we free this memory?
    configureKernel cuNet
    let nsteps = 1000
    sim <- initMemory cuNet att nsteps stdpConf
    return $ Simulation nsteps
        (stepMulti sim nsteps probeIdx probeF dt)
        (withForeignPtr (rt sim) Kernel.elapsedMs)
        (withForeignPtr (rt sim) Kernel.resetTimer)
        (getWeights' sim)
        (deviceDiagnostics (rt sim))
        (return ()) -- foreign pointer finalizers clean up
    where
        getWeights' sim = do
            ns <- Memory.getWeights sim
            return $! N.Network ns (Node 0) -- dummy topology



-------------------------------------------------------------------------------
-- Running the simulation
-------------------------------------------------------------------------------


sequence3_ m (a:as) (b:bs) (c:cs) = m a b c >> sequence3_ m as bs cs
sequence3_ _ _  _ _ = return ()

-- TODO: limit probe to specific neurons
stepMulti
    :: SimData
    -> Int                  -- ^ simulation steps
    -> Probe
    -> Maybe (ProbeFn IzhState)
    -> TemporalResolution   -- ^ temporal subresolution
    -> [[Idx]]
    -> [STDPApplication]
    -> IO [ProbeData]
stepMulti sim steps pidx pfn dt fstim stdpApp = do
    sequence3_ (stepOne sim pidx pfn dt) fstim [0..] stdpApp
    withForeignPtr (rt sim) $ \rtPtr -> do
    printCycleCounters rtPtr
    case pfn of
        Nothing -> do
            syncSimulation rtPtr
            return $ replicate steps $ FiringData []
        Just Firing -> do
            (_, fired) <- readFiring $ rt sim
            return $ densifyDeviceFiring (att sim) steps fired
        Just ProbeFiringCount -> do
            nfired <- readFiringCount $ rt sim
            return $ [FiringCount nfired]
        _ -> fail $ "CUDA.stepMulti: unsupported probe: " ++ show pfn


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


stepOne sim probeIdx probeF tempSubres fstim currentCycle stdpApplication = do
    let flen = length fstim
        fbounds = (0, flen-1)
    fstim <- forM fstim $ rethrow "firing stimulus" $ deviceIdxM (att sim)
    fsPIdxArr <- newListArray fbounds (map (fromIntegral . partitionIdx) fstim)
    fsNIdxArr <- newListArray fbounds (map (fromIntegral . neuronIdx) fstim)
    withStorableArray fsPIdxArr  $ \fsPIdxPtr -> do
    withStorableArray fsNIdxArr  $ \fsNIdxPtr -> do
    withForeignPtr (rt sim)      $ \rtPtr     -> do
    kernelStatus <- c_step currentCycle
        (fromIntegral tempSubres)
        applySTDP stdpReward
        (fromIntegral flen) fsPIdxPtr fsNIdxPtr
        rtPtr
    when (kernelStatus /= 0) $ fail "Backend error"

    where

        -- TODO: just use a maybe type here instead, unwrap in KernelFFI
        (applySTDP, stdpReward) = case stdpApplication of
            STDPIgnore    -> (0, 0)
            (STDPApply r) -> (1, realToFrac r)
