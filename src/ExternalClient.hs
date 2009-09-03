{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{- | Run nemo through an external thrift-based client interface -}

module ExternalClient (runExternalClient) where

import Control.Concurrent (forkIO)
import Control.Exception
import Control.Monad (liftM4, forever, when)
import Control.Parallel.Strategies (rnf)
import Data.Maybe (fromJust)
import Data.IORef
import qualified Data.Map as Map (Map, mapWithKey)
import Data.Maybe (isJust, fromJust)
import Data.Typeable (Typeable)
import Network
import Thrift
import Thrift.Protocol.Binary
import Thrift.Transport.Handle
import System.Exit (exitSuccess)

-- imports from auto-generated modules. These are not hierarchical
import NemoFrontend hiding (process)
import NemoFrontend_Iface
import qualified Nemo_Types as Wire

-- TODO: just import Construction base module
import qualified Construction.Network as Network
import Construction.Neurons hiding (addNeuron)
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
    = Constructing !Net !Config
    | Simulating !Config !Backend.Simulation


data Config = Config {
        stdpConfig :: !StdpConf,
        simConfig :: !SimulationOptions
    }


defaultConfig = Config (defaults stdpOptions) (defaults $ simOptions AllBackends)


{- | Create the initials state for construction -}
initNemoState :: NemoState
initNemoState = Constructing Network.empty defaultConfig


type ClientState = IORef NemoState


{- Apply function as long as we're in constuction mode -}
constructWith :: IORef NemoState -> (Net -> Net) -> IO ()
constructWith m f = do
    st <- readIORef m
    case st of
        Constructing net conf -> do
            net' <- return $! f net
            st' <- return $! Constructing net' conf
            net' `seq` st' `seq` writeIORef m st'
        _ -> fail "trying to construct during simulation"



{- Modify configuration, regardles of mode -}
reconfigure :: IORef NemoState -> (Config -> Config) -> IO ()
reconfigure m f = do
    st <- readIORef m
    case st of
        Constructing net conf -> do
            let st' = Constructing net (f conf)
            st' `seq` writeIORef m st'
        Simulating conf sim -> do
            let st' = Simulating (f conf) sim
            st' `seq` writeIORef m st'



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

simulateWithLog :: String -> IORef NemoState -> (Backend.Simulation -> IO a) -> IO a
simulateWithLog fn m f = putStrLn fn >> simulateWith m f


simulateWith :: IORef NemoState -> (Backend.Simulation -> IO a) -> IO a
simulateWith m f = do
    st <- readIORef m
    case st of
        Simulating _ sim -> f sim
        Constructing net conf -> do
            -- Start simulation first
            handle initError $ do
            sim <- initSim net (simConfig conf) cudaOpts (stdpConfig conf)
            Backend.start sim
            ret <- f sim
            st' <- return $! Simulating conf sim
            st' `seq` writeIORef m st'
            return $! ret
    where
        -- TODO: make this backend options instead of cudaOpts
        -- TODO: perhaps we want to be able to send this to server?
        cudaOpts = defaults cudaOptions

        initError :: SomeException -> IO a
        initError e = do
            putStrLn $ "initialisation error: " ++ show e
            throwIO $ Wire.ConstructionError $ Just $ show e




clientStopSimulation :: IORef NemoState -> IO ()
clientStopSimulation m = do
    st <- readIORef m
    case st of
        Simulating conf sim -> do
            Backend.stop sim
            writeIORef m $! initNemoState
        -- Simulating net conf sim -> do
            -- Backend.stop sim
            -- return $! (Constructing net conf)
        c@(Constructing _ _) -> return ()


clientReset :: IORef NemoState -> IO ()
clientReset m = do
    putStrLn "resetting state"
    clientStopSimulation m
    writeIORef m $! initNemoState



instance NemoFrontend_Iface ClientState where
    -- TODO: handle Maybes here!
    setBackend h (Just host) = reconfigure h $ setHost host
    addNeuron h (Just n) = constructWith h $ clientAddNeuron n
    run h (Just stim) = simulateWith h $ Protocol.run stim
    enableStdp h (Just prefire) (Just postfire) (Just maxWeight) =
        reconfigure h $ setStdpFn prefire postfire maxWeight
    disableStdp h = reconfigure h $ disableStdp'
    applyStdp h (Just reward) = simulateWithLog "apply STDP" h (\s -> Backend.applyStdp s reward)
    getConnectivity h = simulateWith h Protocol.getConnectivity
    startSimulation h = simulateWithLog "start" h (\_ -> return ())
    stopSimulation h = clientStopSimulation h
    terminate h = clientStopSimulation h >> throw ClientTermination
    reset = clientReset



-- TODO: handle errors here, perhaps just handle in constructWith?
clientAddNeuron :: Wire.IzhNeuron -> Net -> Net
clientAddNeuron wn net = rnf n `seq` Network.addNeuron idx n net
    where
        (idx, n) = Protocol.decodeNeuron wn


process handler (iprot, oprot) = do
    (name, typ, seqid) <- readMessageBegin iprot
    proc handler (iprot,oprot) (name,typ,seqid)
    return True


{- | A binary non-threaded server, which only deals with requests from a single
 - external client -}
runThreadedServer
    :: (Transport t, Protocol i, Protocol o)
    => (Socket -> IO (i t, o t))
    -> (IORef NemoState -> (i t, o t) -> IO Bool)
    -> PortID
    -> IO a
runThreadedServer accepter proc port = do
    socket <- listenOn port
    forever $ do
        ps <- (accepter socket)
        st <- newIORef initNemoState
        loop $ (proc st) ps
    where
        loop m = do { continue <- m; when continue (loop m) }


runExternalClient :: IO ()
runExternalClient = do
    -- TODO: deal with other exceptions as well
    -- Control.Exception.handle (\(TransportExn s t) -> fail s) $ do
    Control.Exception.handle (\ClientTermination -> exitSuccess) $ do
    runThreadedServer binaryAccept process (PortNumber 56101)
    where
        binaryAccept s = do
            (h, _, _) <- accept s
            return (BinaryProtocol h, BinaryProtocol h)
