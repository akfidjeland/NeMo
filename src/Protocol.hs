{- Encoding and decoding between wire data format and memory data format -}

module Protocol (
        defaultPort,
        run,
        getConnectivity,
        decodeNeuron,
        encodeNeuron,
        decodeStdpConfig,
        encodeStimulus,
        decodeFiring,
        decodeConnectivity)
    where

import Control.Parallel.Strategies (rnf)
import Data.Maybe (fromJust)
import qualified Data.Map as Map (Map, mapWithKey)
import Network (PortID(PortNumber))

import qualified Nemo_Types as Wire

import Construction.Neuron (Neuron, neuron, synapses, ndata)
import Construction.Izhikevich (IzhNeuron(..))
import Construction.Synapse (Synapse(..), Static(..), current)
import qualified Simulation as Backend (Simulation, Simulation_Iface(..))
import Simulation.STDP (StdpConf(..))
import Types (Idx, FiringOutput(..))


defaultPort = PortNumber 56100



-- TODO: propagate errors to external client
run :: [Wire.Stimulus] -> Backend.Simulation -> IO [Wire.Firing]
run stimulus sim =
    return . map decodeFiring =<< Backend.run sim (map decodeStimulus stimulus)


getConnectivity :: Backend.Simulation -> IO (Map.Map Idx [Wire.Synapse])
getConnectivity sim = return . encodeConnectivity =<< Backend.getWeights sim


{- | Convert neuron from wire format to internal format -}
decodeNeuron :: Wire.IzhNeuron -> (Idx, Neuron (IzhNeuron Double) Static)
decodeNeuron wn = (idx, neuron n ss)
    where
        idx = fromJust $! Wire.f_IzhNeuron_index wn
        n = IzhNeuron
                (fromJust $! Wire.f_IzhNeuron_a wn)
                (fromJust $! Wire.f_IzhNeuron_b wn)
                (fromJust $! Wire.f_IzhNeuron_c wn)
                (fromJust $! Wire.f_IzhNeuron_d wn)
                (fromJust $! Wire.f_IzhNeuron_u wn)
                (fromJust $! Wire.f_IzhNeuron_v wn)
                0.0 False Nothing
        ss = map (decodeSynapse idx) $! fromJust $! Wire.f_IzhNeuron_axon wn


{- | Convert neuron from internal format to wire format -}
encodeNeuron :: (Idx, Neuron (IzhNeuron Double) Static) -> Wire.IzhNeuron
encodeNeuron (idx, n) = Wire.IzhNeuron (Just idx) a b c d u v ss
    where
        p f = Just $! f $! ndata n
        a = p paramA
        b = p paramB
        c = p paramC
        d = p paramD
        u = p stateU
        v = p stateV
        ss = Just $ map encodeSynapse $ synapses idx n


{- | Convert synapse from wire format to internal format -}
decodeSynapse :: Idx -> Wire.Synapse -> Synapse Static
decodeSynapse src ws = Synapse src tgt d $! Static w
    where
        tgt = fromJust $! Wire.f_Synapse_target ws
        d = fromJust $! Wire.f_Synapse_delay ws
        w = fromJust $! Wire.f_Synapse_weight ws


encodeSynapse :: Synapse Static -> Wire.Synapse
encodeSynapse s = Wire.Synapse tgt d w
    where
        tgt = Just $ target s
        d   = Just $ delay s
        w   = Just $ current $ sdata s


decodeStimulus :: Wire.Stimulus -> [Idx]
decodeStimulus = maybe (fail "invalid firing stimulus") id . Wire.f_Stimulus_firing


encodeStimulus :: [Idx] -> Wire.Stimulus
encodeStimulus = Wire.Stimulus . Just


decodeFiring :: FiringOutput -> Wire.Firing
decodeFiring (FiringOutput xs) = xs


decodeConnectivity :: Map.Map Int [Wire.Synapse] -> Map.Map Idx [Synapse Static]
decodeConnectivity = Map.mapWithKey (\i ns -> map (decodeSynapse i) ns)


encodeConnectivity :: Map.Map Idx [Synapse Static] -> Map.Map Int [Wire.Synapse]
encodeConnectivity = fmap (fmap encodeSynapse)


decodeStdpConfig :: [Double] -> [Double] -> Double -> StdpConf
decodeStdpConfig prefire postfire maxWeight =
    StdpConf True prefire postfire maxWeight Nothing
