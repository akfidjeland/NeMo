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
import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Array
import Foreign.Marshal.Utils (toBool)
import Foreign.Ptr
import Foreign.Storable
import Network.Socket

import Construction.Neuron
import qualified Construction.Neurons as Neurons (fromList)
import Construction.Izhikevich
import Construction.Network
import Construction.Synapse
import Construction.Topology
import Network.SocketSerialisation
import Network.Protocol (startSimulation, stopSimulation, runSimulation, getWeights, defaultPort)
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
    -> Ptr CDouble      -- ^ lookup table for potentiation
    -> Ptr CDouble      -- ^ lookup table for depression
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
        useSTDP tp td pot dep mw
        a b c d u v sidx sdelay sweight = do
    potLUT <- peekArray (fromIntegral tp) pot
    depLUT <- peekArray (fromIntegral td) dep
    net <- createNetwork n spn a b c d u v sidx sdelay sweight
    sock <- socketFD fd
    handle (\_ -> return False) $ do
    startSimulation sock net (fromIntegral tsr) (stdpConf potLUT depLUT)
    return True
    where
        -- TODO: might want to control regular STDP application from the host
        stdpConf potLUT depLUT = STDPConf {
                stdpEnabled      = toBool useSTDP,
                stdpPotentiation = map realToFrac potLUT,
                stdpDepression   = map realToFrac depLUT,
                stdpMaxWeight    = (realToFrac mw),
                stdpFrequency    = Nothing
            }



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
    -> Ptr (Ptr CInt)   -- ^ firing (cycles) (caller should free)
    -> Ptr (Ptr CInt)   -- ^ firing (indices) (caller should free)
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



{- | Return weights from the running simulation.
 -
 - The format of the returned data is the same as in the connectivity matrix
 - specification in hs_startSimulation.
 -
 - The weights may change slightly in the process of setting up the simulation,
 - as the backend may use a different floating point precision than the caller.
 -
 - The order of the weight will almost certainly be different from the
 - specification in hs_startSimulation, but each call to hs_getWeights should
 - return the synapses in the same order.
 -
 - The caller should free the allocated data.
 -}
foreign export ccall hs_getWeights
    :: CInt              -- ^ socket file descriptor
    -> Ptr (Ptr CInt)    -- ^ targets
    -> Ptr (Ptr CUInt)   -- ^ delays
    -> Ptr (Ptr CDouble) -- ^ weights
    -> Ptr CUInt         -- ^ number of neurons
    -> Ptr CUInt         -- ^ number of synapses per neurons
    -> IO Bool           -- ^ success
hs_getWeights fd t_ptr d_ptr w_ptr nc_ptr sc_ptr = do
    handle (\_ -> return False) $ do
    ns <- getWeights =<< socketFD fd
    let (_, max) = idxBounds ns
    {- TODO: the caller should already know this, so no need to compute it
     - again every time. This should be only instance of maxSynapsesPerNeuron
     - so remove dead code if changing to allocation in caller -}
    let pitch = maxSynapsesPerNeuron ns
    let size  = pitch * (max+1)
    targets <- newArray $ replicate size nullNeuron
    delays  <- mallocArray size
    weights <- mallocArray size
    mapM_ (pokeNeuron ns pitch targets delays weights) [0..max]
    poke t_ptr targets
    poke d_ptr delays
    poke w_ptr weights
    poke nc_ptr $ fromIntegral $ max+1
    poke sc_ptr $ fromIntegral pitch
    return True



pokeNeuron
    :: Network Stateless Static
    -> Int
    -> Ptr CInt
    -> Ptr CUInt
    -> Ptr CDouble
    -> Int
    -> IO ()
pokeNeuron ns pitch t_ptr d_ptr w_ptr n_idx = do
    let ss = synapsesOf ns n_idx
    let i = rowOffset n_idx
    pokeSynapses (plusPtr t_ptr (i*4)) (plusPtr d_ptr (i*4)) (plusPtr w_ptr (i*8)) ss

    where
        rowOffset n = pitch * n

        pokeSynapses :: Ptr CInt -> Ptr CUInt -> Ptr CDouble -> [Synapse Static] -> IO ()
        pokeSynapses _ _ _ [] = return ()
        pokeSynapses t_ptr d_ptr w_ptr (s:ss) = do
            poke t_ptr $ fromIntegral $ target s
            poke d_ptr $ fromIntegral $ delay s
            poke w_ptr $ realToFrac $ current $ sdata s
            -- TODO: detect row overflow?
            pokeSynapses
                (plusPtr t_ptr 4)
                (plusPtr d_ptr 4)
                (plusPtr w_ptr 8) ss



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
    return $! Network (Neurons.fromList ns)  t
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
peekAxon
    :: Idx
    -> Int
    -> Int
    -> Ptr CInt
    -> Ptr CInt
    -> Ptr CDouble
    -> IO [Synapse Static]
peekAxon source i m t d w = go i (i+m)
    where
        go i end
            | i == end  = return []
            | otherwise = do
                s  <- peekSynapse i source t d w
                ss <- go (i+1) end
                case s of
                    Nothing -> return $! ss -- don't assume all null synapses at end of row
                    Just s  -> return $! (s:ss)


{- | Synapses pointing to the null neuron are considered inactive -}
nullNeuron = -1

{- | Get a single synapse out of c-array -}
peekSynapse
    :: Int
    -> Idx
    -> Ptr CInt
    -> Ptr CInt
    -> Ptr CDouble
    -> IO (Maybe (Synapse Static))
peekSynapse i source t_arr d_arr w_arr = do
    target <- peekElemOff t_arr i
    delay  <- peekElemOff d_arr i
    weight <- peekElemOff w_arr i
    if target == nullNeuron
        then return $! Nothing
        else return $! Just $!
                Synapse source (fromIntegral target) (fromIntegral delay) $!
                    Static (realToFrac weight)


foreign export ccall hs_defaultPort :: IO CUShort
hs_defaultPort = return $ fromIntegral defaultPort
