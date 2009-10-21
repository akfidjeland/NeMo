{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE BangPatterns #-}

module Server (runServer, Duration(..)) where

import Control.Concurrent (forkOS)
import qualified Control.Exception as CE
import Control.Monad (when, forever, replicateM_)
import Control.Parallel.Strategies (rnf)
import Data.IORef
import Data.List (foldl')
import Data.Typeable (Typeable)
import Network (PortID(..), HostName, accept, listenOn)
import Network.Socket.Internal (PortNumber)
import Thrift
import Thrift.Protocol.Binary
import Thrift.Transport.Handle
-- TODO: undo!
-- import System.IO (Handle, hPutStrLn)
import System.IO
import System.Time (getClockTime)

import Construction.Izhikevich (IzhNeuron)
import qualified Construction.Network as Network (Network, addNeuron,
        addNeuronGroup, empty, hPrintConnections, idxBounds)
import Construction.Synapse (Static)
import qualified Simulation as Backend (Simulation, Simulation_Iface(..))
import Simulation.Pipelined (initSim)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.Options (SimulationOptions(..), Backend(..), defaultBackend)
import Simulation.STDP (StdpConf(..))
import Options (defaults)
import qualified Protocol (decodeNeuron,
        run, getConnectivity, defaultPort, decodeStdpConfig, decodeStimulus,
        pipelineLength)
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
        stdpConfig :: StdpConf,
        pipelined :: Bool
    }


newConfig :: StdpConf -> Config
newConfig s = Config s False -- pipelining disabled by default



{- Apply function as long as we're in constuction mode -}
constructWith :: Handler -> (Net -> Net) -> IO ()
constructWith (Handler _ m) f = do
    st <- readIORef m
    case st of
        Constructing conf net -> do
            net' <- return $! f net
            st' <- return $! Constructing conf net'
            net' `seq` st' `seq` writeIORef m st'
        _ -> fail "trying to construct during simulation"


-- TODO: share with frontend code
serverAddNeuron :: Wire.IzhNeuron -> Net -> Net
serverAddNeuron wn net = n `seq` Network.addNeuron idx n net
    where
        (idx, n) = Protocol.decodeNeuron wn


serverAddCluster :: Handler -> [Wire.IzhNeuron] -> IO ()
serverAddCluster (Handler log m) ns = do
    log "add cluster"
    st <- readIORef m
    case st of
        (Simulating _) -> CE.throwIO $! Wire.ConstructionError $!
            Just $! "simulation command called before construction complete"
        (Constructing conf !net) -> do
            net' <- return $! foldl' (flip serverAddNeuron) net ns
            st' <- return $! Constructing conf net'
            writeIORef m st'


configureWithLog :: Handler -> String -> (Config -> Config) -> IO ()
configureWithLog h@(Handler log _) fnName f = log fnName >> configureWith h f


configureWith :: Handler -> (Config -> Config) -> IO ()
configureWith (Handler _ m) f = do
    st <- readIORef m
    case st of
        Constructing conf net -> writeIORef m $! Constructing (f conf) net
        _ -> fail "Configuration commands must be called /before/ starting simulation"


serverEnableStdp :: Handler -> [Double] -> [Double] -> Double -> Double -> IO ()
serverEnableStdp h@(Handler log m) prefire postfire mxw mnw = do
    let stdp = Protocol.decodeStdpConfig prefire postfire mxw mnw
    configureWithLog h "enabling STDP" $ \conf -> conf { stdpConfig = stdp }



simulateWithLog h@(Handler log mvar) fnName f = log fnName >> simulateWith h f


simulateWith :: Handler -> (Backend.Simulation -> IO a) -> IO a
simulateWith (Handler _ m) f = do
    st <- readIORef m
    case st of
        Simulating sim -> do
            ret <- f sim
            writeIORef m st
            return $! ret
        Constructing conf net -> do
            CE.handle initError $ do
            file <- openFile "cm.dat" WriteMode
            Network.hPrintConnections file net
            hClose file
            sim <- initSim net (simConfig conf) cudaOpts (stdpConfig conf)
            ret <- f sim
            writeIORef m $! Simulating sim
            return $! ret
    where

        -- TODO: make this backend options instead of cudaOpts
        -- TODO: perhaps we want to be able to send this to server?
        simConfig conf = SimulationOptions Forever 4 defaultBackend $ pipelined conf
        cudaOpts = defaults cudaOptions

        initError :: CE.SomeException -> IO a
        initError e = do
            putStrLn $ "initialisation error: " ++ show e
            CE.throwIO $ Wire.ConstructionError $ Just $ show e



serverStopSimulation :: Handler -> IO ()
serverStopSimulation (Handler log m) = do
    log "stopping simulation"
    st <- readIORef m
    case st of
        Simulating sim -> Backend.stop sim
        -- TODO: should this be an error?
        Constructing _ _ -> return ()


data Handler = Handler {
        log :: String -> IO (),
        state :: IORef ServerState
    }

instance NemoBackend_Iface Handler where
    -- TODO: use constructWith
    -- deal with Maybes here. Server can crash if client sends garbage or goes down
    addCluster h (Just ns) = serverAddCluster h ns
    addNeuron h (Just n) = constructWith h $ serverAddNeuron n
    startSimulation h = simulateWithLog h "starting simulation" $ const $ return ()
    enableStdp h (Just pre) (Just post) (Just mxw) (Just mnw) =
        serverEnableStdp h pre post mxw mnw
    enablePipelining h =
        configureWithLog h "enabling pipelining" $ \conf -> conf { pipelined = True }
    run h (Just stim) = simulateWith h $ Protocol.run stim
    pipelineLength h = simulateWithLog h "pipeline length query" $ Protocol.pipelineLength

    applyStdp h (Just reward) = simulateWithLog h "apply STDP" (\s -> Backend.applyStdp s reward)
    getConnectivity h = simulateWithLog h "returning connectivity" Protocol.getConnectivity

    {- This is a bit of a clunky way to get out of the serving of a single
     - client. Unfortunately, the auto-generated protocol code does not allow
     - an exit hook. It would be simple in principle (just replace 'process'),
     - but would cause a brittle build. -}
    stopSimulation h = serverStopSimulation h >> CE.throw StopSimulation


data StopSimulation = StopSimulation
    deriving (Show, Typeable)

instance CE.Exception StopSimulation


{- | A binary non-threaded server, which only deals with requests from a single
 - external client at a time. Typically invoked as either 'runServer Forever'
 - or 'runServer Once' (if it should terminate after serving a single client). -}
runServer :: Duration -> Handle -> StdpConf -> PortID -> IO ()
runServer ntimes hdl stdpOptions port = do
    logTime hdl "started"
    let conf = newConfig stdpOptions
    socket <- listenOn port
    -- TODO: forkIO here, keep mvar showing simulation in use
    repeated ntimes $ do
        (h, client, clientPort) <- accept socket
        let log = logMsg hdl client clientPort
        log "connected"
        let ps = (BinaryProtocol h, BinaryProtocol h)
        st <- newIORef $ Constructing conf Network.empty
        CE.handle (\e -> log (show (e::TransportExn)) >> log "disconnected") $ do
        CE.handle (\StopSimulation -> log "disconnected") $ do
        loop $ process (Handler log st) ps
    logTime hdl "terminated"
    where
        loop m = do { continue <- m; when continue (loop m) }


repeated :: Duration -> IO () -> IO ()
repeated Forever = forever
repeated (Until t) = replicateM_ t
repeated Once = id


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
