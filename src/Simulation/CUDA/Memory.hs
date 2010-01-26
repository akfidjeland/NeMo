{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}

module Simulation.CUDA.Memory (
    initMemory,
    getWeights
) where



import Control.Monad (zipWithM_, forM_)
import Data.Array.MArray (newListArray)
import Data.Maybe (Maybe, isNothing)
import qualified Data.Map as Map (Map, fromList, empty)
import Foreign.C.Types
import Foreign.Marshal.Array (mallocArray)
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr
import Foreign.Storable (pokeElemOff)

import Construction.Neuron (terminalsByDelay, Neuron(..))
import Construction.Network (Network, toList)
import Construction.Izhikevich (IzhNeuron(..), stateSigma)
import Construction.Synapse (AxonTerminal(AxonTerminal), Static(..), target)
import Simulation.CUDA.Address
import Simulation.CUDA.KernelFFI
import Simulation.CUDA.State (State(..))
import Simulation.CUDA.Mapping
import Simulation.STDP
import Types
import Util.List (maximumM)




{- Initialise memory on a single device -}
initMemory
    :: CuNet IzhNeuron Static
    -> Network IzhNeuron Static
    -> ATT
    -> Int
    -> Int
    -> StdpConf
    -> IO State
initMemory net fullnet att maxProbePeriod dt stdp = do
    (pcount, psizes, maxDelay, rt) <- allocRT net maxProbePeriod
    configureStdp rt stdp
    loadAllNeurons rt net
    setNeurons rt $ toList fullnet
    loadCMatrix rt att net
    copyToDevice rt
    return $ State pcount psizes maxDelay (fromIntegral dt) att rt


loadPartitionNeurons
    :: Ptr CuRT
    -> CSize
    -> CuPartition IzhNeuron Static
    -> IO ()
loadPartitionNeurons rt pidx partition = do
    forPartition loadA $ listOf paramA
    forPartition loadB $ listOf paramB
    forPartition loadC $ listOf paramC
    forPartition loadD $ listOf paramD
    forPartition loadU $ listOf initU
    forPartition loadV $ listOf initV
    let sigma = allJust 0 $ listOf stateSigma
    maybe (return ()) (forPartition loadThalamicInputSigma) sigma
    where
        listOf f = map f ns
        forPartition load xs = do
            arr <- newListArray (0, len-1) $ map realToFrac xs
            load rt pidx len arr
        ns = neurons partition
        -- TODO: just read the partition size
        len = length ns


loadAllNeurons :: Ptr CuRT -> CuNet IzhNeuron Static -> IO ()
loadAllNeurons rt net = zipWithM_ (loadPartitionNeurons rt) [0..] $ partitions net


setNeurons :: Ptr CuRT -> [(Idx, Neuron IzhNeuron Static)] -> IO ()
setNeurons rt ns = mapM_ (setOne rt) ns
    where
        setOne rt (idx, neuron) = do
            let n = ndata neuron
                sigma = maybe 0 id $ stateSigma n
            addNeuron rt idx
                (paramA n) (paramB n) (paramC n) (paramD n)
                (initU n) (initV n) sigma



{- | Return just a list with Nothing replaced by default value. If all are
 - nothing, return Nothing -}
allJust :: a -> [Maybe a] -> Maybe [a]
allJust d xs
    | all isNothing xs = Nothing
    | otherwise = Just $ map toDefault xs
    where
        toDefault Nothing = d
        toDefault (Just x) = x


-------------------------------------------------------------------------------
-- Connectivity matrix
-------------------------------------------------------------------------------


{- | By the time we write data to the device, we have already established the
 - maximum pitch for the connectivity matrices. The data is written on a
 - per-row basis. To avoid excessive memory allocation we allocate a single
 - buffer (for each CM) with the know maximum pitch, and re-use it for each
 - row. -}
data Outbuf = Outbuf {
        weights :: Ptr CFloat,
        nidx    :: Ptr CUInt,
        plasticity :: Ptr CUChar
    }


allocOutbuf len = do
    wbuf <- mallocArray len
    nbuf <- mallocArray len
    spbuf <- mallocArray len
    return $! Outbuf wbuf nbuf spbuf


{- | Write all connectivity data to device -}
loadCMatrix rt att net = do
    -- TODO: what we really want here is a dynamically resized bit of memory
    -- which we allocate only once.
    bufL0 <- allocOutbuf $ 2^16
    forM_ (partitionAssocs net) $ \(pidx, p) -> do
        forM_ (neuronAssocs p) $ \(nidx, n) -> do
            forM_ (terminalsByDelay n) $ \(delay, ss) -> do
                len0 <- pokeSynapses bufL0 0 ss
                -- TODO: don't convert to L0 or L1 here
                -- TODO: remove redundant translation back and forth
                setRow rt bufL0 (globalIdx att (pidx, nidx)) delay len0
    where
        -- TODO: just send global neuron indices here
        -- TODO: remove pidx from data type
        setRow rt buf = setCMDRow rt (weights buf) (nidx buf) (plasticity buf)




{- | Write a row of synapses (i.e for a single presynaptic/delay) to output
 - buffer -}
pokeSynapses _ len0 [] = return len0
pokeSynapses buf0 i0 (s:ss) = do
    pokeSynapse buf0 i0 s
    pokeSynapses buf0 (i0+1) ss


{- | Write a single synapse to output buffer -}
pokeSynapse :: Outbuf -> Int ->  (Idx, Current, Bool, Static) -> IO ()
pokeSynapse buf i (target, weight, plastic, _) = do
    pokeElemOff (weights buf) i $! realToFrac $! weight
    pokeElemOff (nidx buf) i $! fromIntegral $! target
    pokeElemOff (plasticity buf) i $! fromBool plastic


type DArr = (Ptr CInt, Ptr CInt, Ptr CFloat, Int)

pitch :: DArr -> Int
pitch (_, _, _, p) = p


{- | Get (possibly modified) connectivity matrix back from device -}
getWeights :: State -> IO (Map.Map Idx [AxonTerminal Static])
getWeights sim = (return . Map.fromList) =<< mapM (getNWeights sim) (deviceIndices (att sim))


-- return data for a single neuron (all delays)
getNWeights :: State -> DeviceIdx -> IO (Idx, [AxonTerminal Static])
getNWeights sim sdidx = do
    sgidx <- globalIdxM (att sim) sdidx
    ss <- (return . concat) =<< mapM (getNDWeights sim sdidx) [1..maxDelay sim]
    return $! (sgidx, ss)


-- return data for a single neuron (single delay)
getNDWeights :: State -> DeviceIdx -> Delay -> IO [AxonTerminal Static]
getNDWeights sim source d = do
    let sp = partitionIdx source
    let sn = neuronIdx source
    ws <- getCMDRow (rt sim) sp sn d
    return $! map pack $ ws
    where
        pack :: (DeviceIdx, Weight, Bool) -> AxonTerminal Static
        pack (didx, w, plastic) =
            AxonTerminal (globalIdx (att sim) didx) d w plastic ()


-------------------------------------------------------------------------------
-- Runtime data
-------------------------------------------------------------------------------

-- TODO: move to KernelFFI
foreign import ccall unsafe "allocRuntimeData"
    c_allocRT
        :: CSize  -- ^ partition count
        -> CSize  -- ^ max partition size
        -> CUInt  -- ^ set reverse matrix (bool)
        -> CUInt  -- ^ max read period
        -> IO (Ptr CuRT)


allocRT :: CuNet n s -> Int -> IO (Int, [Int], Delay, Ptr CuRT)
allocRT net maxProbePeriod = do
    let pcount = partitionCount net
        psizes = partitionSizes net
        dmax   = maxNetworkDelay net
    rt <- c_allocRT
        (fromIntegral pcount)
        (fromIntegral $! either error id $ maximumM psizes)
        (fromBool $ usingStdp net)
        (fromIntegral maxProbePeriod)
    return $! (pcount, psizes, dmax, rt)
