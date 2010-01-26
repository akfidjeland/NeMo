{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}

module Simulation.CUDA.Memory (
    initMemory,
    getWeights
) where



import Control.Monad (zipWithM_, forM_)
import Data.Maybe (Maybe, isNothing)
import qualified Data.Map as Map (Map, fromList, empty)
import Foreign.C.Types
import Foreign.Marshal.Array (mallocArray)
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr
import Foreign.Storable (pokeElemOff)

import Construction.Axon (terminalsUnordered)
import Construction.Neuron (terminalsByDelay, Neuron(..))
import Construction.Network (Network, toList)
import Construction.Izhikevich (IzhNeuron(..), stateSigma)
import Construction.Synapse (AxonTerminal(AxonTerminal), Static(..),
    plastic, target, weight, delay)
import Simulation.CUDA.Address
import Simulation.CUDA.KernelFFI
import Simulation.CUDA.State (State(..))
import Simulation.CUDA.Mapping
import Simulation.STDP
import Types
import Util.List (maximumM)




{- Initialise memory on a single device -}
initMemory
    :: Network IzhNeuron Static
    -> ATT
    -> Maybe Int -- ^ requested partition size
    -> Int
    -> Int
    -> StdpConf
    -> IO State
initMemory fullnet att reqPsize maxProbePeriod dt stdp = do
    rt <- allocRT stdp reqPsize maxProbePeriod
    configureStdp rt stdp
    setNeurons rt $ toList fullnet
    copyToDevice rt
    return $ State (fromIntegral dt) att rt



setNeurons :: Ptr CuRT -> [(Idx, Neuron IzhNeuron Static)] -> IO ()
setNeurons rt ns = do
    buf <- allocOutbuf $ 2^16
    mapM_ (setOne rt buf) ns
    where
        setOne rt buf (idx, neuron) = do
            let n = ndata neuron
                sigma = maybe 0 id $ stateSigma n
            addNeuron rt idx
                (paramA n) (paramB n) (paramC n) (paramD n)
                (initU n) (initV n) sigma
            let ss = terminalsUnordered $ axon neuron
            len <- pokeSynapses0 buf 0 ss
            addSynapses rt idx
                (nidx buf) (delays buf) (weights buf) (plasticity buf) len


{- | Write a row of synapses (i.e for a single presynaptic/delay) to output
 - buffer -}
pokeSynapses0 :: Outbuf -> Int -> [AxonTerminal Static] -> IO Int
pokeSynapses0 _ len0 [] = return len0
pokeSynapses0 buf0 i0 (s:ss) = do
    pokeSynapse0 buf0 i0 s
    pokeSynapses0 buf0 (i0+1) ss


{- | Write a single synapse to output buffer -}
pokeSynapse0 :: Outbuf -> Int -> AxonTerminal Static -> IO ()
pokeSynapse0 buf i s = do
    pokeElemOff (weights buf) i $! realToFrac $! weight s
    pokeElemOff (nidx buf) i $! fromIntegral $! target s
    pokeElemOff (plasticity buf) i $! fromBool $! plastic s
    pokeElemOff (delays buf) i $! fromIntegral $! delay s



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
        plasticity :: Ptr CUChar,
        delays :: Ptr CUInt
    }


allocOutbuf len = do
    wbuf <- mallocArray len
    nbuf <- mallocArray len
    spbuf <- mallocArray len
    dbuf <- mallocArray len
    return $! Outbuf wbuf nbuf spbuf dbuf


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
    -- TODO: fix this, without relying on knowing delays on this side of API
    error "need to implement getNWeights"
    -- sgidx <- globalIdxM (att sim) sdidx
    -- ss <- (return . concat) =<< mapM (getNDWeights sim sdidx) [1..maxDelay sim]
    -- return $! (sgidx, ss)


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


allocRT :: StdpConf -> Maybe Int -> Int -> IO (Ptr CuRT)
allocRT stdp reqPsize maxProbePeriod = do
    psize <- targetPartitionSize reqPsize
    rt <- allocateRuntime
        psize
        (stdpEnabled stdp)
        maxProbePeriod
    return rt



{- | The partition size the mapper should use depends on both the maximum
 - supported by the kernel (given some configuration) and the request of the
 - user. partitionSize determines this size and logs a message if the user
 - request is over-ridden.  -}
targetPartitionSize :: Maybe Int -> IO Int
targetPartitionSize userSz = do
    let maxSz = maxPartitionSize
    maybe (return maxSz) (validateUserSz maxSz) userSz
    where
        validateUserSz maxSz userSz =
            if userSz <= maxSz
                then return userSz
                else do
                    putStrLn $ "Invalid partition size (" ++ show userSz
                        ++ ") requested" ++ ", defaulting to " ++ show maxSz
                    return maxSz
