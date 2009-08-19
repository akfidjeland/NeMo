{- Run simulation on a remote host.
 -
 - The network is built on the client and the client provides the stimulus,
 - while the server maintains the simulation. -}

module Network.Client (initSim) where

import Control.Parallel.Strategies (NFData)
import Control.Exception (handle, SomeException)
import Data.Binary (Binary)
import Network.Socket

import Construction.Network (Network)
import qualified Network.Protocol as Wire (startSimulation,
           runSimulation, stopSimulation, applyStdp, getWeights)
import Simulation (Simulation_Iface(..))
import Simulation.FiringStimulus (denseToSparse)
import Simulation.STDP (StdpConf)
import Types


data RemoteSimulation = RSim Socket


instance Simulation_Iface RemoteSimulation where
    run (RSim s) = runRemote s
    step _ = error "single step not supported in client backend"
    applyStdp (RSim s) reward = Wire.applyStdp s reward
    elapsed _ = error "timing functions not supported in client backend"
    resetTimer _ = error "timing functions not supported in client backend"
    getWeights (RSim s) = Wire.getWeights s
    terminate (RSim s) = Wire.stopSimulation s >> sClose s



-- TODO: refactor wrt runSim in CUDA and CPU
-- | Return a step function which forwards execution to remote host
initSim :: (Binary n, Binary s, NFData n, NFData s)
    => String         -- ^ hostname
    -> Int            -- ^ port number
    -> Network n s
    -> TemporalResolution
    -> StdpConf
    -> IO RemoteSimulation
initSim hostname port net dt stdpConf = do
    sock <- openSocket hostname (show port)
    handle (initError sock) $ do
    Wire.startSimulation sock net dt stdpConf
    return $! RSim sock
    where
        initError :: Socket -> SomeException -> IO RemoteSimulation
        initError s e = sClose s >> fail (show e)



-- | Step through a fixed number of simulation cycles on the remote host
runRemote :: Socket -> [[Idx]] -> IO [ProbeData]
runRemote sock fstim = do
    (firing, _) <- Wire.runSimulation sock nsteps $ denseToSparse fstim
    return $ map FiringData firing
    where
        nsteps = length fstim



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
