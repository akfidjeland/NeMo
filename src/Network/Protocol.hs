module Network.Protocol (
    -- * Client interface
    startSimulation, runSimulation, stopSimulation, applyStdp, getWeights,
    -- * Host interface
    startSimulationHost,
    -- TODO: move both sides of protocol to this fil
    recvRequest, ClientRequest(..),
    sendResponse, ServerResponse(..),
    recvCommand, ClientCommand(..),
    -- * Common interface
    defaultPort,
    SimulationInit
) where

import Control.Monad
import Control.Parallel.Strategies (NFData, rnf, using)
import Data.Binary
import qualified Data.Map as Map (Map)
import Network.Socket

import Construction.Network (Network)
import Construction.Synapse (Synapse, Static)
import Network.SocketSerialisation (sendSerialised, recvSerialised)
import Simulation (Simulation)
import Simulation.STDP
import Types (Time, TemporalResolution, Idx)

defaultPort :: Int
defaultPort = 56100


type SimulationInit n s
    =  Network n s
    -> TemporalResolution
    -> StdpConf
    -> IO Simulation


-- Starting the simulation

-- | Send start request from client and check return status
startSimulation
    :: (Binary n, Binary s, NFData n, NFData s)
    => Socket
    -> Network n s
    -> TemporalResolution
    -> StdpConf
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
            return $! Just stepfn
        ReqPing      -> sendResponse sock RspReady >> return Nothing
        (ReqError c) -> fail $ "Invalid start request: " ++ show c


-- | Send data request from client and return data from host
-- TODO: just send dense data here
runSimulation :: Socket -> Time -> [(Time, [Idx])] -> IO ([[Idx]], Int)
runSimulation sock nsteps fstim = do
    sendCommand sock $ CmdSync nsteps fstim
    rsp <- recvResponse sock
    case rsp of
        RspData d elapsed -> return (d, elapsed)
        RspError msg      -> fail $ "runSimulation: " ++ msg
        _                 -> fail "runSimulation: unexpected response"


{- | Send shutdown instruction from client to host -}
stopSimulation :: Socket -> IO ()
stopSimulation sock = sendCommand sock CmdStop


applyStdp :: Socket -> Double -> IO ()
applyStdp sock reward = sendCommand sock $! CmdApplyStdp reward


{- | Request weights from host -}
getWeights :: Socket -> IO (Map.Map Idx [Synapse Static])
getWeights sock = do
    sendCommand sock CmdGetWeights
    rsp <- recvResponse sock
    case rsp of
        RspWeights ns -> return ns
        RspError msg  -> fail $ "getWeights: " ++ msg
        _             -> fail "getWeights: unexpected response"



-- simulation setup

sendRequest
    :: (Binary n, Binary s, NFData n, NFData s)
    => Socket -> ClientRequest n s -> IO ()
sendRequest = sendSerialised

recvRequest
    :: (Binary n, Binary s, NFData n, NFData s)
    => Socket -> IO (ClientRequest n s)
recvRequest = recvSerialised

data ClientRequest n s
        = ReqStart !(Network n s) TemporalResolution StdpConf
        -- = ReqStart (Network n s) TemporalResolution (Maybe StdpConf)
        | ReqPing
        | ReqError Word8
    deriving (Eq)

instance (Binary n, Binary s, NFData n, NFData s) => Binary (ClientRequest n s) where
    put (ReqStart n tr stdp) = putWord8 1 >> put n >> put tr >> put stdp
    put ReqPing              = putWord8 2
    put (ReqError _)         = putWord8 0
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
        = CmdSync Time [(Time, [Idx])]
        | CmdStop
        | CmdGetWeights       -- ^ return full weight matrix
        | CmdApplyStdp Double -- ^ apply STDP with the given reward
        | CmdError Word8
    deriving (Show, Eq)


instance Binary ClientCommand where
    put (CmdSync duration f) = putWord8 3 >> put duration >> put f
    put CmdStop = putWord8 4
    put CmdGetWeights = putWord8 5
    put (CmdApplyStdp reward) = putWord8 6 >> put reward
    put (CmdError _) = putWord8 0
    get = do
        tag <- getWord8
        case tag of
            3 -> liftM2 CmdSync get get
            4 -> return CmdStop
            5 -> return CmdGetWeights
            6 -> liftM CmdApplyStdp get
            _ -> return $ CmdError tag


sendResponse :: Socket -> ServerResponse -> IO ()
sendResponse = sendSerialised

recvResponse :: Socket -> IO ServerResponse
recvResponse = recvSerialised

data ServerResponse
        = RspStart
        -- ^ dense firing data + number of milliseconds of simulation
        | RspData [[Idx]] Int
        | RspError String
        | RspReady
        | RspBusy
        | RspWeights (Map.Map Idx [Synapse Static])

instance Binary ServerResponse where
    put = putRsp
    get = getRsp

putRsp :: ServerResponse -> Put
putRsp RspStart                 = putWord8 0
putRsp (RspData firing elapsed) = putWord8 1 >> put firing >> put elapsed
putRsp (RspError err)           = putWord8 2 >> put err
putRsp RspReady                 = putWord8 3
putRsp RspBusy                  = putWord8 4
putRsp (RspWeights net)         = putWord8 5 >> put net

getRsp :: Get ServerResponse
getRsp = do
    tag <- getWord8
    case tag of
        0 -> return RspStart
        1 -> liftM2 RspData get get
        2 -> liftM RspError get
        3 -> return RspReady
        4 -> return RspBusy
        5 -> liftM RspWeights get
        _ -> fail "Decoding server response failed"
