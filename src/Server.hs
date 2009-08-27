{-# LANGUAGE FlexibleInstances #-}

module Server (runServer, forever, once) where

import Control.Concurrent.MVar
import qualified Control.Exception as CE
import Control.Monad (when, forever)
import Control.Parallel.Strategies (rnf)
import Data.Typeable (Typeable)
import qualified Data.Map as Map (Map, mapWithKey)
import Network (PortID(..), HostName, accept, listenOn)
import Network.Socket.Internal (PortNumber)
import Thrift
import Thrift.Protocol.Binary
import Thrift.Transport.Handle
import System.IO (Handle, hPutStrLn)
import System.Time (getClockTime)

import Construction.Izhikevich (IzhNeuron, IzhState)
import qualified Construction.Network as Network (Network, addNeuron, addNeuronGroup, empty)
import Construction.Synapse (Static)
import qualified Simulation as Backend (Simulation, Simulation_Iface(..))
import Simulation.Backend (initSim)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.Options (SimulationOptions(..), Backend(..))
import Simulation.STDP (StdpConf(..))
import Options (defaults)
import qualified Protocol (decodeNeuron, run, getConnectivity, defaultPort, decodeStdpConfig)
import Types

import NemoBackend
import NemoBackend_Iface
import qualified Nemo_Types as Wire

type Net = Network.Network (IzhNeuron Double) Static

{- | The server currently only deals with a single simulation at a time. This
 - goes through two stages: construction and simulation -}
data ServerState
        = Constructing Config Net
        | Simulating Backend.Simulation


data Config = Config {
        stdpConfig :: StdpConf
    }


{- Apply function as long as we're in constuction mode -}
constructWith :: Handler -> (Net -> Net) -> IO ()
constructWith (Handler _ m) f = do
    modifyMVar_ m $ \st -> do
    case st of
        Constructing conf net -> do
            net' <- return $! f net
            return $! Constructing conf net'
        _ -> fail "trying to construct during simulation"


-- TODO: share with frontend code
serverAddNeuron :: Wire.IzhNeuron -> Net -> Net
serverAddNeuron wn net = rnf n `seq` Network.addNeuron idx n net
    where
        (idx, n) = Protocol.decodeNeuron wn


serverAddCluster :: Handler -> [Wire.IzhNeuron] -> IO ()
serverAddCluster (Handler log mvar) ns = do
    log "add cluster"
    modifyMVar_ mvar $ \st -> do
    case st of
        (Simulating _) -> CE.throwIO $! Wire.ConstructionError $!
            Just $! "simulation command called before construction complete"
        (Constructing conf net) -> do
            let ns' = fmap Protocol.decodeNeuron ns
            return $! Constructing conf $! Network.addNeuronGroup ns' net


serverFinaliseNetwork :: Handler -> IO ()
serverFinaliseNetwork (Handler log mvar) = do
    log "finalise network"
    modifyMVar_ mvar $ \st -> do
    case st of
        -- TODO: throw exception, user might want a warning
        s@(Simulating _) -> return s
        Constructing conf net -> do
            sim <- initSim net simConfig cudaOpts (stdpConfig conf)
            return $! Simulating sim
    where
        simConfig = SimulationOptions Forever 4 CUDA
        cudaOpts = defaults cudaOptions


serverEnableStdp :: Handler -> [Double] -> [Double] -> Double -> IO ()
serverEnableStdp (Handler log m) prefire postfire mw = do
    log "enabling STDP"
    modifyMVar_ m $ \st -> do
    case st of
        Constructing conf net -> do
            let conf' = Config $ Protocol.decodeStdpConfig prefire postfire mw
            return $! Constructing conf' net
        _ -> fail "STDP must be enabled /before/ starting simulation"


simulateWith :: Handler -> (Backend.Simulation -> IO a) -> IO a
simulateWith (Handler log mvar) f = do
    log "sim"
    modifyMVar mvar $ \st -> do
    case st of
        Simulating sim -> do
            ret <- f sim
            return $! (st, ret)
        Constructing conf net -> do
            -- Start simulation first
            CE.handle initError $ do
            sim <- initSim net simConfig cudaOpts (stdpConfig conf)
            ret <- f sim
            return $! (Simulating sim, ret)
    where
        -- TODO: make this backend options instead of cudaOpts
        -- TODO: perhaps we want to be able to send this to server?
        simConfig = SimulationOptions Forever 4 CUDA
        cudaOpts = defaults cudaOptions

        initError :: CE.SomeException -> IO (ServerState, a)
        initError e = do
            putStrLn $ "initialisation error: " ++ show e
            CE.throwIO $ Wire.ConstructionError $ Just $ show e



serverStopSimulation :: Handler -> IO ()
serverStopSimulation (Handler log m) = do
    log "stopping simulation"
    withMVar m $ \st -> do
    case st of
        Simulating sim -> Backend.terminate sim
        -- TODO: should this be an error?
        Constructing _ _ -> return ()


data Handler = Handler {
        log :: String -> IO (),
        state :: MVar ServerState
    }

instance NemoBackend_Iface Handler where
    -- TODO: handle maybes
    -- TODO: use constructWith
    addCluster st (Just ns) = serverAddCluster st ns
    addNeuron h (Just n) = constructWith h $ serverAddNeuron n
    finaliseNetwork = serverFinaliseNetwork
    enableStdp st (Just pre) (Just post) (Just mw) = serverEnableStdp st pre post mw
    run st (Just stim) = simulateWith st $ Protocol.run stim
    applyStdp st (Just reward) = simulateWith st (\s -> Backend.applyStdp s reward)
    getConnectivity st = simulateWith st Protocol.getConnectivity

    {- This is a bit of a clunky way to get out of the serving of a single
     - client. Unfortunately, the auto-generated protocol code does not allow
     - an exit hook. It would be simple in principle (just replace 'process'),
     - but would cause a brittle build. -}
    stopSimulation st = serverStopSimulation st >> CE.throw StopSimulation


data StopSimulation = StopSimulation
    deriving (Show, Typeable)

instance CE.Exception StopSimulation


{- | A binary non-threaded server, which only deals with requests from a single
 - external client at a time. Typically invoked as either 'runServer forever'
 - or 'runServer once' (if it should terminate after serving a single client). -}
runServer :: (IO () -> IO ()) -> Handle -> StdpConf -> PortID -> IO ()
runServer ntimes hdl stdpOptions port = do
    logTime hdl "started"
    let conf = Config stdpOptions
    socket <- listenOn port
    -- TODO: forkIO here, keep mvar showing simulation in use
    ntimes $ do
        (h, client, clientPort) <- accept socket
        let log = logMsg hdl client clientPort
        log "connected"
        let ps = (BinaryProtocol h, BinaryProtocol h)
        st <- newMVar $ Constructing conf Network.empty
        CE.handle (\e -> log (show (e::TransportExn)) >> log "disconnected") $ do
        CE.handle (\StopSimulation -> log "disconnected") $ do
        loop $ process (Handler log st) ps
    logTime hdl "terminated"
    where
        loop m = do { continue <- m; when continue (loop m) }


once :: IO () -> IO ()
once = id


{- | Log message, with time prepended -}
logTime :: Handle -> String -> IO ()
logTime hdl msg = do
    t <- getClockTime
    hPutStrLn hdl $ show t ++ " " ++ msg


{- | Log message along with client information -}
logMsg :: Handle -> HostName -> PortNumber -> String -> IO ()
logMsg hdl addr pn msg = do
    let source = "From " ++ addr ++ ":" ++ show pn ++ ": "
    logTime hdl $ source ++ msg
