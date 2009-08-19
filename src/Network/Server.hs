module Network.Server (serveSimulation) where

-- Based on example in Real World Haskell, Chapter 27

import Control.Exception (try, handle, IOException)
import Control.Parallel.Strategies (NFData)
import Data.Binary (Binary)
import Data.List (sort)
import Network.Socket
import System.IO (Handle, hPutStrLn)
import System.Time (getClockTime)

import Control.Exception(assert)

import Network.Protocol hiding (getWeights, applyStdp)
import Simulation (Simulation_Iface(..))
import Simulation.STDP
import Types
import qualified Util.Assocs as A (densify)
import qualified Util.List as L (chunksOf)



serveSimulation
    :: (Show n, Binary n, Show s, Binary s, NFData n, NFData s)
    => Handle           -- ^ Log output
    -> String           -- ^ Port number or name
    -> Bool             -- ^ Verbose
    -> SimulationInit n s
    -> IO ()
serveSimulation loghdl port verbose initfn = withSocketsDo $ do

    -- Look up the port. This either raises an exception or returns a non-empty
    -- list
    addrinfos <- getAddrInfo
        (Just (defaultHints {
                addrFlags = [AI_PASSIVE],
                addrFamily = AF_UNSPEC }))
        Nothing
        (Just port)
    let serveraddr = head addrinfos

    sock <- socket (addrFamily serveraddr) Stream defaultProtocol

    -- TODO: remove in production code. This makes TCP less reliable
    setSocketOption sock ReuseAddr 1

    bindSocket sock (addrAddress serveraddr)

    -- start listening for connection requests
    let maxQueueLength = 5
    listen sock maxQueueLength
    -- create a lock to use for synchronising access to the handler
    -- lock <- newMVar ()

    -- loop forever waiting for connections. Ctrl-C to abort
    procRequests sock loghdl verbose initfn



-- | Process incoming connection requests (only handle one at a time)
-- TODO: add back handling of multiple requests, but make sure to send
-- back the correct status code if busy
procRequests
    :: (Binary n, Show n, Binary s, Show s, NFData n, NFData s)
    => Socket
    -> Handle
    -> Bool
    -> SimulationInit n s
    -> IO ()
procRequests mastersock hdl verbose initfn = do
    (connsock, clientaddr) <- accept mastersock
    logMsg hdl clientaddr "client connected"
    (catch
        (procSim connsock verbose initfn (logMsg hdl clientaddr))
        (\e -> logMsg hdl clientaddr $
            "exception caught, simulation terminated\n\t" ++ show e))
    sClose connsock
    logMsg hdl clientaddr "client disconnected"
    procRequests mastersock hdl verbose initfn



{- | Process potential simulation request -}
procSim sock _ initfn log = do
    handle initError $ do
    ret <- startSimulationHost sock initfn
    case ret of
        Nothing  -> return () -- ping or similar
        Just sim -> procSimReq sock sim log
    where
        initError :: IOException -> IO ()
        initError e = log $ show e



-- | Process user requests during running simulation
-- TODO: move some of this to Protocol.hs
procSimReq sock sim log = do
    req <- recvCommand sock
    case req of
        (CmdSync nsteps fstim) -> do
            -- TODO: re-order arguments
            withSim sock (procSynReq nsteps sim fstim) $ do
                (\(probed, elapsed) -> sendResponse sock $ RspData probed elapsed)
        CmdStop  -> stop sim
        CmdGetWeights -> do
            log "returning weight matrix"
            weights <- getWeights sim
            sendResponse sock $ RspWeights weights
            procSimReq sock sim log

        (CmdApplyStdp reward) -> withSim sock ((applyStdp sim) reward) (\_ -> return ())

        (CmdError c) -> do
            log $ "invalid simulation request: " ++ show c
            stop sim
    where
        stop sim = terminate sim >> log "stopping simulation"

        withSim :: Socket -> IO a -> (a -> IO ()) -> IO ()
        withSim sock simFn outFn =
            try simFn >>= either
                (\err -> do
                    sendResponse sock $ RspError $ show (err :: IOException)
                    log $ "error: " ++ show err
                    dmsg <- diagnostics sim
                    log $ "diagnostics:\n" ++ dmsg
                    stop sim)
                (\a -> outFn a >> procSimReq sock sim log)


procSynReq nsteps sim sparseFstim = do
    resetTimer sim
    probed <- run sim fstim
    e <- elapsed sim
    putStrLn $ "Simulated " ++ (show nsteps) ++ " steps in " ++ (show e) ++ "ms"
    assert ((length probed) == nsteps) $ do
    return (map getFiring probed, fromIntegral e)
    where
        fstim = A.densify 0 nsteps [] sparseFstim

        -- | We only deal with firing data here
        getFiring (NeuronState _) = error "Server.hs: simulation returned non-firing data"
        getFiring (FiringData firing) = sort firing


-- | Log message along with client information
logMsg :: Handle -> SockAddr -> String -> IO ()
logMsg hdl addr msg = do
    t <- getClockTime
    hPutStrLn hdl $ show t ++ " From " ++ show addr ++ ": " ++ "Server.hs: " ++ msg
