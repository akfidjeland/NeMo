{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}

-- TODO: merge whole file with KernelFFI?

module Simulation.CUDA.Memory (
    initSim,
    getWeights,
    State(..)
) where


import Control.Monad (when, foldM)
import Data.Maybe (Maybe)
import qualified Data.Map as Map (Map, fromList)
import qualified Data.Set as Set (Set, empty, insert, toList)
import Foreign.C.Types
import Foreign.Marshal.Array (mallocArray)
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr
import Foreign.Storable (pokeElemOff)

import Construction.Axon (terminalsUnordered)
import Construction.Neuron (Neuron(..))
import qualified Construction.Network as Network (Network, toList)
import Construction.Izhikevich (IzhNeuron(..), stateSigma)
import Construction.Synapse (AxonTerminal(AxonTerminal), Static(..),
    plastic, target, weight, delay)
import Data.Set
import Simulation.CUDA.Address
import Simulation.CUDA.KernelFFI
import Simulation.STDP
import Types


data State = State {
        rtdata :: Ptr CuRT,     -- ^ kernel runtime data
        indices :: Set.Set Idx -- ^ all neurons in the network
    }

{- Initialise memory on a single device -}
initSim
    :: Network.Network IzhNeuron Static
    -> Maybe Int -- ^ requested partition size
    -> StdpConf
    -> IO State
initSim net reqPsize stdp = do
    rt <- allocateRuntime reqPsize (stdpEnabled stdp)
    when (rt == nullPtr) $ fail "Failed to create CUDA simulation"
    configureStdp rt stdp
    indices <- setNeurons rt $ Network.toList net
    initSimulation rt
    return $ State rt indices



setNeurons :: Ptr CuRT -> [(Idx, Neuron IzhNeuron Static)] -> IO (Set.Set Idx)
setNeurons rt ns = do
    buf <- allocOutbuf $ 2^16
    foldM (setOne rt buf) Set.empty ns
    where
        setOne rt buf indices (idx, neuron) = do
            let n = ndata neuron
                sigma = maybe 0 id $ stateSigma n
            addNeuron rt idx
                (paramA n) (paramB n) (paramC n) (paramD n)
                (initU n) (initV n) sigma
            let ss = terminalsUnordered $ axon neuron
            len <- pokeSynapses buf 0 ss
            addSynapses rt idx
                (nidx buf) (delays buf) (weights buf) (plasticity buf) len
            return $! Set.insert idx indices


{- | Write a row of synapses (i.e for a single presynaptic/delay) to output
 - buffer -}
pokeSynapses :: Outbuf -> Int -> [AxonTerminal Static] -> IO Int
pokeSynapses _ len0 [] = return len0
pokeSynapses buf0 i0 (s:ss) = do
    pokeSynapse buf0 i0 s
    pokeSynapses buf0 (i0+1) ss


{- | Write a single synapse to output buffer -}
pokeSynapse :: Outbuf -> Int -> AxonTerminal Static -> IO ()
pokeSynapse buf i s = do
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
 - buffer with the know maximum pitch, and re-use it for each row. -}
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


{- | Get (possibly modified) connectivity matrix back from device -}
getWeights :: State -> IO (Map.Map Idx [AxonTerminal Static])
getWeights sim = do
    let idxs = Set.toList $ indices sim
    axons <- mapM (getNWeights sim) idxs
    return $! Map.fromList $ zip idxs axons


-- return data for a single neuron (single delay)
getNWeights :: State -> Idx -> IO [AxonTerminal Static]
getNWeights sim source = do
    ws <- getSynapses (rtdata sim) source
    return $! fmap pack $ ws
    where
        pack :: (Idx, Delay, Weight, Bool) -> AxonTerminal Static
        pack (idx, d, w, plastic) = AxonTerminal idx d w plastic ()
