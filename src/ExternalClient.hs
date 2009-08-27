{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{- | Run nemo through an external thrift-based client interface -}

module ExternalClient (runExternalClient) where

import Control.Concurrent (forkIO)
import Control.Concurrent.MVar
import Control.Exception
import Control.Monad (liftM4, forever, when)
import Control.Parallel.Strategies (rnf)
import Data.Maybe (fromJust)
import qualified Data.Map as Map (Map, mapWithKey)
import Data.Typeable (Typeable)
import Network
import Thrift
import Thrift.Protocol.Binary
import Thrift.Transport.Handle
import System.Exit (exitSuccess)

-- imports from auto-generated modules. These are not hierarchical
import NemoFrontend
import NemoFrontend_Iface
import qualified Nemo_Types as Wire

-- TODO: just import Construction base module
import qualified Construction.Network as Network
import Construction.Neurons
import Construction.Neuron
import Construction.Synapse
import Construction.Izhikevich
import Construction.Topology
import Options
import qualified Simulation as Backend (Simulation, Simulation_Iface(..))
import Simulation.CUDA.Options
import Simulation.Options
import Simulation.Backend (initSim)
import Simulation.STDP
import Simulation.STDP.Options
import qualified Protocol (decodeNeuron, run, getConnectivity, decodeStdpConfig, defaultPort)
import Types


type Net = Network.Network (IzhNeuron Double) Static

data ClientException = ClientTermination
    deriving (Show, Typeable)

instance Exception ClientException

-- TODO: add user configuration options

data NemoState
    = Constructing Net Config
    | Simulating Net Config Backend.Simulation


data Config = Config {
        stdpConfig :: StdpConf,
        simConfig :: SimulationOptions
    }


defaultConfig = Config (defaults stdpOptions) (defaults $ simOptions AllBackends)


{- | Return the /static/ network -}
network :: NemoState -> Net
network (Constructing net _) = net
network (Simulating net _ _) = net


{- | Create the initials state for construction -}
initNemoState :: NemoState
initNemoState = Constructing Network.empty defaultConfig


type ClientState = MVar NemoState



{- Apply function as long as we're in constuction mode -}
constructWith :: MVar NemoState -> (Net -> Net) -> IO ()
constructWith m f = do
    modifyMVar_ m $ \st -> do
    case st of
        Constructing net conf -> do
            net' <- return $! f net
            return $! Constructing net' conf
        _ -> fail "trying to construct during simulation"


{- Modify configuration, regardles of mode -}
reconfigure :: MVar NemoState -> (Config -> Config) -> IO ()
reconfigure m f = do
    modifyMVar_ m $ \st -> do
    case st of
        Constructing net conf -> return $! Constructing net (f conf)
        Simulating net conf sim -> return $! Simulating net (f conf) sim



setStdpFn :: [Double] -> [Double] -> Double -> Config -> Config
setStdpFn prefire postfire maxWeight conf = conf { stdpConfig = stdpConfig' }
    where
        stdpConfig' = Protocol.decodeStdpConfig prefire postfire maxWeight


disableStdp' :: Config -> Config
disableStdp' conf = conf { stdpConfig = stdpConfig' }
    where
        stdpConfig' = (stdpConfig conf) { stdpEnabled = False }


setHost :: String -> Config -> Config
setHost host conf = conf { simConfig = simConfig' }
    where
        simConfig' = (simConfig conf) {
                optBackend = RemoteHost host Protocol.defaultPort
            }




simulateWith :: MVar NemoState -> (Backend.Simulation -> IO a) -> IO a
simulateWith m f = do
    modifyMVar m $ \st -> do
    case st of
        Simulating _ _ sim -> do
            ret <- f sim
            return $! (st, ret)
        Constructing net conf -> do
            -- Start simulation first
            putStr "starting simulation..."
            handle initError $ do
            sim <- initSim net (simConfig conf) cudaOpts (stdpConfig conf)
            ret <- f sim
            putStrLn "done"
            return $! (Simulating net conf sim, ret)
    where
        -- TODO: make this backend options instead of cudaOpts
        -- TODO: perhaps we want to be able to send this to server?
        cudaOpts = defaults cudaOptions

        initError :: SomeException -> IO (NemoState, a)
        initError e = do
            putStrLn $ "initialisation error: " ++ show e
            throwIO $ Wire.ConstructionError $ Just $ show e



clientStopSimulation :: MVar NemoState -> IO ()
clientStopSimulation m = do
    modifyMVar_ m $ \static -> do
    case static of
        Simulating net conf sim -> do
            Backend.terminate sim
            return $! (Constructing net conf)
        c@(Constructing _ _) -> return $! c


clientReset :: MVar NemoState -> IO ()
clientReset m = do
    putStrLn "resetting state"
    clientStopSimulation m
    modifyMVar_ m $ \_ -> return $! initNemoState



instance NemoFrontend_Iface ClientState where
    -- TODO: handle Maybes here!
    setBackend h (Just host) = reconfigure h $ setHost host
    addNeuron h (Just n) = constructWith h $ clientAddNeuron n
    run h (Just stim) = simulateWith h $ Protocol.run stim
    enableStdp h (Just prefire) (Just postfire) (Just maxWeight) =
        reconfigure h $ setStdpFn prefire postfire maxWeight
    disableStdp h = reconfigure h $ disableStdp'
    applyStdp h (Just reward) = simulateWith h (\s -> Backend.applyStdp s reward)
    getConnectivity h = simulateWith h Protocol.getConnectivity
    stopSimulation h = clientStopSimulation h
    reset = clientReset
    terminate h = clientStopSimulation h >> throw ClientTermination



-- TODO: handle errors here, perhaps just handle in constructWith?
clientAddNeuron :: Wire.IzhNeuron -> Net -> Net
clientAddNeuron wn net = rnf n `seq` Network.addNeuron idx n net
    where
        (idx, n) = Protocol.decodeNeuron wn



{- | A binary non-threaded server, which only deals with requests from a single
 - external client -}
runThreadedServer
    :: (Transport t, Protocol i, Protocol o)
    => (Socket -> IO (i t, o t))
    -> h
    -> (h -> (i t, o t) -> IO Bool)
    -> PortID
    -> IO a
runThreadedServer accepter hand proc port = do
    socket <- listenOn port
    forever $ do
        ps <- (accepter socket)
        loop $ (proc hand) ps
    where
        loop m = do { continue <- m; when continue (loop m) }


runExternalClient :: IO ()
runExternalClient = do
    st <- newMVar initNemoState
    -- TODO: deal with other exceptions as well
    -- Control.Exception.handle (\(TransportExn s t) -> fail s) $ do
    Control.Exception.handle (\ClientTermination -> exitSuccess) $ do
    runThreadedServer binaryAccept st process (PortNumber 56101)
    where
        binaryAccept s = do
            (h, _, _) <- accept s
            return (BinaryProtocol h, BinaryProtocol h)
