{- Simulation running remotely on machine connected via socket and using a
 - thrift protocol for communication -}

module Simulation.Remote (initSim) where

import Control.Monad (when)
import Data.Maybe (fromJust)
import qualified Data.Map as Map (Map)
import Data.Word (Word16)
import Network (HostName, PortID(..))
import Network.Socket.Internal (PortNumber(..))
import Thrift.Protocol.Binary (BinaryProtocol(..))
import Thrift.Transport.Handle (hOpen, tClose)
import System.IO (Handle)

import Construction.Izhikevich (IzhNeuron)
import Construction.Synapse (AxonTerminal, Static)
import qualified Construction.Network as Network (Network, toList)
import Protocol (decodeFiring, encodeStimulus, encodeNeuron, decodeConnectivity)
import Simulation (Simulation_Iface(..), Simulation(..))
import Simulation.STDP (StdpConf(..))
import Simulation.Options (SimulationOptions, optPipelined)
import Types (Idx, FiringOutput(..))

import qualified Nemo_Types as Wire
import qualified NemoBackend_Client as Wire


data Remote = Remote {
        to :: Handle,
        ps :: (BinaryProtocol Handle, BinaryProtocol Handle)
    }

type Net = Network.Network (IzhNeuron Double) Static

initSim :: HostName -> PortID -> Net -> SimulationOptions -> StdpConf -> IO Remote
initSim hostName portID net simOpts stdp = do
    to <- hOpen (hostName, portID)
    let p = BinaryProtocol to
    let ps = (p,p)
    when (stdpEnabled stdp) $
        Wire.enableStdp ps (prefire stdp) (postfire stdp)
            (stdpMaxWeight stdp) (stdpMinWeight stdp)
    when (optPipelined simOpts) $ Wire.enablePipelining ps
    sendChunks ps 128 $ map encodeNeuron $ Network.toList net
    -- sendEach ps $ map encodeNeuron $ Network.toList net
    Wire.startSimulation ps
    return $! Remote to ps
    where
        sendChunks _ _ [] = return ()
        sendChunks ps sz ns = do
            let (h, t) = splitAt sz ns
            Wire.addCluster ps h
            sendChunks ps sz t

        sendEach _ [] = return ()
        sendEach ps (n:ns) = do
            Wire.addNeuron ps n
            sendEach ps ns


instance Simulation_Iface Remote where
    run = remoteRun
    step = remoteStep
    pipelineLength = remotePipelineLength
    applyStdp = remoteApplyStdp
    elapsed = error "timing is not supported on remote backend"
    resetTimer = error "timing is not supported on remote backend"
    getWeights = remoteGetWeights
    start r = Wire.startSimulation (ps r)
    stop r = Wire.stopSimulation (ps r)


remoteEnableStdp :: Remote -> StdpConf -> IO ()
remoteEnableStdp r stdp = do
    when (stdpEnabled stdp) $ do
    Wire.enableStdp (ps r) (prefire stdp) (postfire stdp)
        (stdpMaxWeight stdp) (stdpMinWeight stdp)


remoteRun :: Remote -> [[Idx]] -> IO [FiringOutput]
remoteRun r fstim = do
    fired <- Wire.run (ps r) $! map encodeStimulus fstim
    return $! map FiringOutput fired


remoteStep :: Remote -> [Idx] -> IO FiringOutput
remoteStep r fstim = do
    [fired] <- Wire.run (ps r) $! [encodeStimulus fstim]
    return $! FiringOutput fired


remotePipelineLength :: Remote -> IO (Int, Int)
remotePipelineLength r = do
    pll <- Wire.pipelineLength (ps r)
    let input = fromJust $ Wire.f_PipelineLength_input pll
        output = fromJust $ Wire.f_PipelineLength_output pll
    return $! (input, output)


remoteApplyStdp :: Remote -> Double -> IO ()
remoteApplyStdp r reward = Wire.applyStdp (ps r) reward


remoteGetWeights :: Remote -> IO (Map.Map Idx [AxonTerminal Static])
remoteGetWeights r = return . decodeConnectivity =<< Wire.getConnectivity (ps r)
