{- Run simulation on a remote host.
 -
 - The network is built on the client and the client provides the stimulus,
 - while the server maintains the simulation. -}

module Network.Client (initSim) where

import Control.Exception (assert)
import Control.Parallel.Strategies (NFData)
import Data.Binary (Binary)
import Network.Socket

import Construction.Network (Network)
import Network.Protocol
        (startSimulation, runSimulation, stopSimulation, defaultPort)
import Simulation.Common
import Simulation.FiringStimulus (denseToSparse)
import Simulation.STDP (STDPConf, STDPApplication(..))
import Types


-- TODO: refactor wrt runSim in CUDA and CPU
-- | Return a step function which forwards execution to remote host
initSim :: (Binary n, Binary s, NFData n, NFData s)
    => String         -- ^ hostname
    -> Int            -- ^ port number
    -> Network n s
    -> TemporalResolution
    -> STDPConf
    -> IO Simulation
initSim hostname port net dt stdpConf = do
    sock <- openSocket hostname (show port)
    -- TODO: get STDP configuration from options instead
    startSimulation sock net dt stdpConf
    {- To reduce network overheads we deal with one second's worth of data at a
     - time. -}
    -- TODO: might want to adjust this at run-time
    -- TODO: when do we close socket? When Simluation goes out of scope perhaps?
    let stepsz = 1000
    return $ Simulation stepsz
        (stepRemote sock stepsz)
        (return 0)              -- TODO: add timing function here
        (return ())             -- TODO: add timing function here
        -- TODO: add function to forward request for weights
        (error "getWeights not implemented in 'client' backend")
        (closeRemote sock)



-- | Step through a fixed number of simulation cycles on the remote host
stepRemote :: Socket -> Int -> SimulationStep
-- TODO: get STDP application from caller
stepRemote sock stepsz fstim _ = do
    assert (length fstim == stepsz) $ do
    -- Here we only support fixed-rate application of STDP, which is configured during initialisation
    (firing, _) <- runSimulation sock stepsz (denseToSparse fstim) STDPIgnore
    return $ map FiringData firing



closeRemote :: Socket -> IO ()
closeRemote sock = stopSimulation sock >> sClose sock



-- | Open socket given a hostname and a port number or name
openSocket
    :: String    -- ^ Remote hostname, or localhost
    -> String    -- ^ Port number or name
    -> IO Socket
openSocket hostname port = do
    addrinfos <- getAddrInfo
        (Just (defaultHints { addrFamily = AF_INET }))
        (Just hostname)
        (Just port)
    let serveraddr = head addrinfos
    sock <- socket (addrFamily serveraddr) Stream defaultProtocol
    -- connection may be idle for long time, esp. if used interactively
    setSocketOption sock KeepAlive 1
    -- TODO: may want to specify buffering mode
    connect sock (addrAddress serveraddr)
    return sock
