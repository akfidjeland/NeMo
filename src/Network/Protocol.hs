module Network.Protocol (
    -- * Client interface
    startSimulation, runSimulation, stopSimulation,
    -- * Host interface
    startSimulationHost,
    -- TODO: move both sides of protocol to this file
    recvRequest, ClientRequest(..),
    sendResponse, ServerResponse(..),
    recvCommand, ClientCommand(..),
    -- * Common interface
    defaultPort
) where

import Control.Monad
import Control.Parallel.Strategies (NFData, rnf, using)
import Data.Binary
import Network.Socket

import Construction.Network (Network)
import Network.SocketSerialisation (sendSerialised, recvSerialised)
import Simulation.Common (Simulation, SimulationInit)
import Simulation.STDP
import Types (Time, TemporalResolution, Idx)

defaultPort :: Int
defaultPort = 56100


-- Starting the simulation

-- | Send start request from client and check return status
startSimulation
    :: (Binary n, Binary s, NFData n, NFData s)
    => Socket
    -> Network n s
    -> TemporalResolution
    -> Maybe STDPConf
    -> IO ()
startSimulation sock net tr stdpConf = do
    sendRequest sock (ReqStart net tr stdpConf)
    rsp <- recvResponse sock
    case rsp of
        RspStart -> return ()
        _        -> fail "startSimulation: unexpected response"


-- | Return true if a simulation was started
startSimulationHost
    :: (Binary n, Binary s, NFData n, NFData s)
    => Socket
    -> SimulationInit n s
    -> IO (Maybe Simulation)
startSimulationHost sock initfn = do
    req <- recvRequest sock
    case req of
        ReqStart net tr stdp -> do
            -- TODO: catch errors in initfn before responding
            stepfn <- initfn net tr stdp
            sendResponse sock RspStart
            return $ Just stepfn
        ReqPing      -> sendResponse sock RspReady >> return Nothing
        (ReqError c) -> fail $ "Invalid start request: " ++ show c


-- | Send data request from client and return data from host
runSimulation :: Socket -> Time -> [(Time, [Idx])] -> STDPApplication -> IO ([[Idx]], Int)
runSimulation sock nsteps fstim stdp = do
    sendCommand sock $ CmdSync nsteps fstim stdp
    rsp <- recvResponse sock
    case rsp of
        RspData d elapsed -> return (d, elapsed)
        _                 -> fail "runSimulation: unexpected response"


-- | Send shutdown instruction from client to host
stopSimulation :: Socket -> IO ()
stopSimulation sock = sendCommand sock CmdStop


-- simulation setup

sendRequest :: (Binary n, Binary s, NFData n, NFData s) => Socket -> ClientRequest n s -> IO ()
sendRequest = sendSerialised

recvRequest :: (Binary n, Binary s, NFData n, NFData s) => Socket -> IO (ClientRequest n s)
recvRequest = recvSerialised

data ClientRequest n s
        = ReqStart !(Network n s) TemporalResolution (Maybe STDPConf)
        -- = ReqStart (Network n s) TemporalResolution (Maybe STDPConf)
        | ReqPing
        | ReqError Word8
    deriving (Eq)

instance (Binary n, Binary s, NFData n, NFData s) => Binary (ClientRequest n s) where
    put (ReqStart n tr stdp) = putWord8 1 >> put n >> put tr >> put stdp
    put ReqPing = putWord8 2
    put (ReqError _) = putWord8 0
    get = do
        tag <- getWord8
        case tag of
            -- TODO: perhaps use rnf on network here?
            -- 1 -> liftM3 ReqStart get get get
            1 -> do
                net  <- get
                tr   <- get
                stdp <- get
                return $! (ReqStart net tr stdp `using` rnf)
            2 -> return ReqPing
            -- TODO: perhaps return a special error value instead?
            _ -> return $ ReqError tag

instance (NFData n, NFData s) => NFData (ClientRequest n s) where
    rnf (ReqStart net tr stdp) = rnf net `seq` rnf tr -- leave out STDP
    rnf (ReqPing) = ()
    rnf (ReqError e) = ()

-- simulation control

sendCommand :: Socket -> ClientCommand -> IO ()
sendCommand = sendSerialised

recvCommand :: Socket -> IO ClientCommand
recvCommand = recvSerialised


data ClientCommand
        = CmdSync Time [(Time, [Idx])] STDPApplication
        | CmdStop
        | CmdError Word8
    deriving (Show, Eq)


instance Binary ClientCommand where
    put (CmdSync duration f stdp) = putWord8 3 >> put duration >> put f >> put stdp
    put CmdStop = putWord8 4
    put (CmdError _) = putWord8 0
    get = do
        tag <- getWord8
        case tag of
            3 -> liftM3 CmdSync get get get
            4 -> return CmdStop
            _ -> return $ CmdError tag


sendResponse :: Socket -> ServerResponse -> IO ()
sendResponse = sendSerialised

recvResponse :: Socket -> IO ServerResponse
recvResponse = recvSerialised

data ServerResponse
        = RspStart
        -- ^ dense firing data + number of milliseconds of simulation
        | RspData [[Idx]] Int
        | RspError
        | RspReady
        | RspBusy

instance Binary ServerResponse where
    put = putRsp
    get = getRsp

putRsp :: ServerResponse -> Put
putRsp RspStart = putWord8 0
putRsp (RspData firing elapsed) = putWord8 1 >> put firing >> put elapsed
putRsp RspError = putWord8 2
putRsp RspReady = putWord8 3
putRsp RspBusy = putWord8 4

getRsp :: Get ServerResponse
getRsp = do
    tag <- getWord8
    case tag of
        0 -> return RspStart
        1 -> liftM2 RspData get get
        2 -> return RspError
        3 -> return RspReady
        4 -> return RspBusy
