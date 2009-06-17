{-# LANGUAGE ForeignFunctionInterface #-}

{- | C-based foreign-function interface (FFI) for simulation clients. -}

module Network.ClientFFI (
    hs_startSimulation,
    hs_runSimulation,
    hs_stopSimulation,
    hs_defaultPort,
    -- * Internals (for testing)
    createNetwork
) where

import Control.Exception (handle)
import Data.Binary
import qualified Data.Map as Map (fromList)
import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import Network.Socket

import Construction.Neuron
import Construction.Izhikevich
import Construction.Network
import Construction.Synapse
import Construction.Topology
import Network.SocketSerialisation
import Network.Protocol (startSimulation, stopSimulation, runSimulation, defaultPort)
import Simulation.STDP
import Types



{- | Start simulation on host connected through the given socket. Return success -}
foreign export ccall hs_startSimulation
    :: CInt             -- ^ socket file descriptor
    -> CInt             -- ^ number of neurons
    -> CInt             -- ^ number of synapses per neuron (must be the same for all)
    -> CInt             -- ^ temporal subresolution of simulation

    -> CInt             -- ^ bool: use STDP?
    -> CInt             -- ^ STDP tau (max time) for potentiation
    -> CInt             -- ^ STDP tau (max time) for depression
    -> CDouble          -- ^ STDP alpha for potentiation
    -> CDouble          -- ^ STDP alpha for depression
    -> CDouble          -- ^ STDP max weight

    -> Ptr CDouble      -- ^ a
    -> Ptr CDouble      -- ^ b
    -> Ptr CDouble      -- ^ c
    -> Ptr CDouble      -- ^ d
    -> Ptr CDouble      -- ^ u
    -> Ptr CDouble      -- ^ v
    -> Ptr CInt         -- ^ synapse indices
    -> Ptr CInt         -- ^ synapse delays
    -> Ptr CDouble      -- ^ syanapse weights
    -> IO Bool
hs_startSimulation fd n spn tsr
        useSTDP tp td ap ad mw
        a b c d u v sidx sdelay sweight = do
    net <- createNetwork n spn a b c d u v sidx sdelay sweight
    sock <- socketFD fd
    handle (\_ -> return False) $ do
    -- TODO: get STDP configuration here instead
    startSimulation sock net (fromIntegral tsr) stdpConf
    return True
    where
        -- TODO: might want to control regular STDP application from the host
        stdpConf
            | useSTDP == 0 = Nothing
            | otherwise   = Just $ STDPConf
                    (fromIntegral tp) (fromIntegral td)
                    (realToFrac ap) (realToFrac ad)
                    (realToFrac mw) Nothing



{- | Stop simulation on the host, return success status -}
foreign export ccall hs_stopSimulation :: CInt -> IO Bool
hs_stopSimulation fd = do
    handle (\_ -> return False) $ do
    socketFD fd >>= stopSimulation
    return True



{- | Run simulation for a number of cycles. Return success status. -}
foreign export ccall hs_runSimulation
    :: CInt             -- ^ socket descriptor
    -> CInt             -- ^ number of simulation cycles
    -> CUInt            -- ^ apply STDP?
    -> CDouble          -- ^ STDP reward signal
    -> Ptr CInt         -- ^ firing stimulus (cycles)
    -> Ptr CInt         -- ^ firing stimulus (indices)
    -> CInt             -- ^ firing stimulus (length)
    -> Ptr (Ptr CInt)   -- ^ firing (cycles), allocated by caller
    -> Ptr (Ptr CInt)   -- ^ firing (indices), allocated by caller
    -> Ptr CInt         -- ^ firing (length)
    -> Ptr CInt         -- ^ elapsed time (milliseconds)
    -> Ptr (Ptr CChar)  -- ^ pointer to error if call failed (caller should free)
    -> IO Bool
hs_runSimulation fd nsteps applySTDP stdpReward
        fscycles fsidx fslen fcycles fidx flen elapsed errMsg = do
    sock <- socketFD fd
    let nsteps' = fromIntegral nsteps
    fstim <- foreignToFlat fscycles fsidx fslen
    handle (\err -> newCString (show err) >>= poke errMsg >> return False) $ do
    (firing, elapsed') <- runSimulation sock nsteps' (flatToSparse fstim) stdpApplication
    (fcycles', fidx', flen') <-
        flatToForeign $ sparseToFlat $ denseToSparse firing
    poke fcycles fcycles'
    poke fidx fidx'
    poke flen flen'
    poke elapsed $ fromIntegral elapsed'
    return True
    where
        stdpApplication
            | applySTDP == 0 = STDPIgnore
            | otherwise      = STDPApply $ realToFrac stdpReward


foreignToFlat :: Ptr CInt -> Ptr CInt -> CInt -> IO [(Time, Idx)]
foreignToFlat cycles idx len = do
    let len' = fromIntegral len
    cycles' <- peekArray len' cycles
    idx' <- peekArray len' idx
    return $ zip (map fromIntegral cycles') (map fromIntegral idx')


-- TODO: migrate this to Simulation/FiringStimulus
-- pre: input ordered by time
flatToSparse :: [(Time, Idx)] -> [(Time, [Idx])]
flatToSparse [] = []
flatToSparse xs = (t, idx) : flatToSparse xs'
    where
        t = fst $ head xs
        current = (==t) . fst
        xs' = dropWhile current xs
        idx = map snd $ takeWhile current xs


-- TODO: use function from FiringStimulus instead
denseToSparse :: [[Idx]] -> [(Time, [Idx])]
denseToSparse fs = zip [0..] fs

sparseToFlat :: [(Time, [Idx])] -> [(Time, Idx)]
sparseToFlat fs = concatMap expand fs
    where
        expand (t, idx) = zip (repeat t) idx


{- | Note: caller should free the data to avoid space leak -}
flatToForeign :: [(Time, Idx)] -> IO (Ptr CInt, Ptr CInt, CInt)
flatToForeign fs = do
    let len = length fs
    -- TODO: use a single dynamically sized array in an IORef
    cycles <- mallocArray len
    idx <- mallocArray len
    pokeArray cycles $ map (fromIntegral . fst) fs
    pokeArray idx $ map (fromIntegral . snd) fs
    return (cycles, idx, fromIntegral len)




{- | Convert file descriptor (returned by openSocket) back to Socket -}
socketFD :: CInt -> IO Socket
-- TODO: We're not sure about the socket type here!
socketFD fd = mkSocket fd AF_INET Stream defaultProtocol Connected


createNetwork n' m' a b c d u v st sd sw = do
    ns <- go 0 0
    return $! Network (Map.fromList ns)  t
    where
        n = fromIntegral n'
        m = fromIntegral m'
        t  = Cluster $ map Node [0..n-1]

        go n_idx s_idx
            | n_idx == n = return []
            | otherwise = do
                nn <- peekNeuron n_idx s_idx m a b c d u v st sd sw
                ns <- go (n_idx+1) (s_idx+m)
                return $! (n_idx, nn) : ns


{- | Get a single neuron out of c-array -}
peekNeuron
    :: Int -> Int -> Int
    -> Ptr CDouble -> Ptr CDouble
    -> Ptr CDouble -> Ptr CDouble
    -> Ptr CDouble -> Ptr CDouble
    -> Ptr CInt         -- ^ synapse indices
    -> Ptr CInt         -- ^ synapse delays
    -> Ptr CDouble      -- ^ syanapse weights
    -> IO (Neuron (IzhNeuron FT) Static)
peekNeuron n_idx s_idx m a b c d u v st sd sw = do
    a' <- peekElemOff a n_idx
    b' <- peekElemOff b n_idx
    c' <- peekElemOff c n_idx
    d' <- peekElemOff d n_idx
    u' <- peekElemOff u n_idx
    v' <- peekElemOff v n_idx
    let n = mkNeuron2
            (realToFrac a') (realToFrac b')
            (realToFrac c') (realToFrac d')
            (realToFrac u') (realToFrac v') Nothing
    ss <- peekAxon n_idx s_idx m st sd sw
    return $! neuron n ss


{- | Get an axon out of c-array -}
peekAxon :: Idx -> Int -> Int -> Ptr CInt -> Ptr CInt -> Ptr CDouble -> IO [Synapse Static]
peekAxon source i m t d w = go i (i+m)
    where
        go i end
            | i == end  = return []
            | otherwise = do
                s  <- peekSynapse i source t d w
                ss <- go (i+1) end
                return $! (s:ss)


{- | Get a single synapse out of c-array -}
peekSynapse :: Int -> Idx -> Ptr CInt -> Ptr CInt -> Ptr CDouble -> IO (Synapse Static)
peekSynapse i source t_arr d_arr w_arr = do
    target <- peekElemOff t_arr i
    delay  <- peekElemOff d_arr i
    weight <- peekElemOff w_arr i
    return $! Synapse source (fromIntegral target) (fromIntegral delay)
            $! Static (realToFrac weight)


foreign export ccall hs_defaultPort :: IO CUShort
hs_defaultPort = return $ fromIntegral defaultPort
