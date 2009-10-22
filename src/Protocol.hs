{-# LANGUAGE BangPatterns #-}

{- Encoding and decoding between wire data format and memory data format.
 -
 - Note: there are various combiations of 'seq', 'rnf' and 'NFData' here left
 - after attempts to optimise the code. These are probably not needed. -}

module Protocol (
        defaultPort,
        run,
        pipelineLength,
        getConnectivity,
        decodeNeuron,
        encodeNeuron,
        decodeStdpConfig,
        decodeStimulus,
        encodeStimulus,
        decodeFiring,
        decodeConnectivity)
    where

import Control.Parallel.Strategies (rnf)
import Data.Maybe (fromJust)
import qualified Data.Map as Map (Map, mapWithKey)
import Network (PortID(PortNumber))

import qualified Nemo_Types as Wire

import Construction.Neuron (Neuron, neuron, terminalsUnordered, ndata)
import Construction.Izhikevich (IzhNeuron(..))
import Construction.Synapse (AxonTerminal(AxonTerminal), Static, Synaptic(..))
import qualified Simulation as Backend (Simulation, Simulation_Iface(..))
import Simulation.STDP (StdpConf(..))
import Types (Idx, FiringOutput(..))


defaultPort = PortNumber 56100


-- TODO: propagate errors to external client
run :: [Wire.Stimulus] -> Backend.Simulation -> IO [Wire.Firing]
run stimulus sim =
    return . map decodeFiring =<< Backend.run sim (map decodeStimulus stimulus)


pipelineLength :: Backend.Simulation -> IO Wire.PipelineLength
pipelineLength sim = do
    -- the backend may be pipelined as well
    (be_i, be_o) <- Backend.pipelineLength sim
    return $! Wire.PipelineLength (Just $ be_i) (Just $ be_o)


getConnectivity :: Backend.Simulation -> IO (Map.Map Idx [Wire.Synapse])
getConnectivity sim = return . encodeConnectivity =<< Backend.getWeights sim



{- | Convert neuron from wire format to internal format -}
decodeNeuron :: Wire.IzhNeuron -> (Idx, Neuron IzhNeuron Static)
decodeNeuron !wn = ss `seq` (idx, neuron n ss)
    {- note: tried adding 'rnf ss' here, but this does not help memory
     - performance -}
    where
        idx = fromJust $! Wire.f_IzhNeuron_index wn
        n = IzhNeuron
                (fromJust $! Wire.f_IzhNeuron_a wn)
                (fromJust $! Wire.f_IzhNeuron_b wn)
                (fromJust $! Wire.f_IzhNeuron_c wn)
                (fromJust $! Wire.f_IzhNeuron_d wn)
                (fromJust $! Wire.f_IzhNeuron_u wn)
                (fromJust $! Wire.f_IzhNeuron_v wn)
                Nothing
        ss = map' decodeSynapse $! fromJust $! Wire.f_IzhNeuron_axon wn

        {- Performance note: Using a strict map rather than a lazy one brought
         - the transfer time for a network with 3.125M synapses down from 65s
         - to 55s.  Additionally reducing y to normal form did not make any
         - further difference either way. -}
        map' f [] = []
        map' f (x:xs) = let y = f x in y `seq` y : map' f xs



{- | Convert neuron from internal format to wire format -}
encodeNeuron :: (Idx, Neuron IzhNeuron Static) -> Wire.IzhNeuron
encodeNeuron (idx, n) = Wire.IzhNeuron (Just idx) a b c d u v ss
    where
        p f = Just $! f $! ndata n
        a = p paramA
        b = p paramB
        c = p paramC
        d = p paramD
        u = p initU
        v = p initV
        ss = Just $! map encodeSynapse $ terminalsUnordered n


{- | Convert synapse from wire format to internal format -}
decodeSynapse :: Wire.Synapse -> AxonTerminal Static
decodeSynapse ws = AxonTerminal tgt d w p ()
    {- note: tried using bang-pattern on ws. This reduced performance -}
    {- note: tried using 'seq' and 'rnf' on all the AxonTerminal fields here,
     - but this did not help with memory usage.  Time-wise it was much the
     - same. Left code in the cleaner form. -}
    where
        tgt = fromJust $! Wire.f_Synapse_target ws
        d = fromJust $! Wire.f_Synapse_delay ws
        w = fromJust $! Wire.f_Synapse_weight ws
        p = fromJust $! Wire.f_Synapse_plastic ws




encodeSynapse :: AxonTerminal Static -> Wire.Synapse
encodeSynapse s = Wire.Synapse tgt d w p
    {- Note: Tried using rnf/seq on the inputs to Wire.Synapse. This made no
     - difference to encoding performance -}
    where
        tgt = Just $! target s
        d   = Just $! delay s
        w   = Just $! weight s
        p   = Just $! plastic s


decodeStimulus :: Wire.Stimulus -> [Idx]
decodeStimulus = maybe (fail "invalid firing stimulus") id . Wire.f_Stimulus_firing


encodeStimulus :: [Idx] -> Wire.Stimulus
encodeStimulus = Wire.Stimulus . Just


decodeFiring :: FiringOutput -> Wire.Firing
decodeFiring (FiringOutput xs) = xs


decodeConnectivity :: Map.Map Int [Wire.Synapse] -> Map.Map Idx [AxonTerminal Static]
decodeConnectivity = fmap (fmap decodeSynapse)


encodeConnectivity :: Map.Map Idx [AxonTerminal Static] -> Map.Map Int [Wire.Synapse]
encodeConnectivity = fmap (fmap encodeSynapse)


decodeStdpConfig :: [Double] -> [Double] -> Double -> Double -> StdpConf
decodeStdpConfig prefire postfire maxWeight minWeight =
    StdpConf True prefire postfire maxWeight minWeight Nothing
