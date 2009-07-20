{-# LANGUAGE MultiParamTypeClasses #-}

module Construction.Neuron (
    NeuronProbe(..),
    mergeProbeFs,
    -- * Construction
    Neuron,
    ndata,
    unconnected,
    neuron,
    Stateless(..),
    -- * Query
    synapses,
    synapsesByDelay,
    synapseCount,
    targets,
    -- * Modify
    connect,
    connectMany,
    disconnect,
    replaceSynapse,
    maxDelay,
    -- * Traversal
    withTargets,
    withSynapses,
    -- * Pretty-printing
    printConnections
) where

import Control.Monad (liftM2)
import Control.Parallel.Strategies (NFData, rnf)
import Data.Binary
import Data.List
import qualified Data.Map as Map
import Data.Maybe (isJust)

import qualified Construction.Axon as Axon
import Construction.Synapse
import Types
import Util.List (replace)



class NeuronProbe p n f where
    probeFn :: p -> n f -> f


mergeProbeFs :: (NeuronProbe p n f) => [p] -> n f -> [f]
mergeProbeFs ps n = map (\p -> probeFn p n) ps


data Neuron n s = Neuron {
        ndata :: n,
        -- TODO: use IArray
        -- TODO: use Data.Seq
        axon :: Axon.Axon s
    } deriving (Eq)



{- | Create a neuron with no connections -}
unconnected :: n -> Neuron n s
unconnected n = Neuron n Axon.unconnected


{- | Create a neuron from a list of connections -}
neuron :: n -> [Synapse s] -> Neuron n s
neuron n ss = Neuron n $ Axon.fromList ss


{- | Apply function to synapses of a neuron -}
withAxon :: (Axon.Axon s -> Axon.Axon s) -> Neuron n s -> Neuron n s
withAxon f (Neuron n ss) = Neuron n $ f ss


withAxonM
    :: (Monad m)
    => (Axon.Axon s -> m (Axon.Axon s)) -> Neuron n s -> m (Neuron n s)
withAxonM f (Neuron n s) = f s >>= return . Neuron n


{- | Return unordered list of all synapses -}
synapses :: Idx -> Neuron n s -> [Synapse s]
synapses src n = Axon.synapses src $ axon n


synapsesByDelay :: Neuron n s -> [(Delay, [(Idx, s)])]
synapsesByDelay = Axon.synapsesByDelay . axon


synapseCount :: Neuron n s -> Int
synapseCount = Axon.size . axon


{- | Return list of target neurons, including duplicates -}
targets :: Neuron n s -> [Target]
targets = Axon.targets . axon


{- | Add a single synapse to a neuron -}
connect :: Synapse s -> Neuron n s -> Neuron n s
connect s = withAxon (Axon.connect s)


{- | Add multiple synapses to a neuron -}
connectMany :: [Synapse s] -> Neuron n s -> Neuron n s
connectMany ss = withAxon (Axon.connectMany ss)


{- | Disconnect the first matching synapse -}
disconnect :: (Eq s) => Synapse s -> Neuron n s -> Neuron n s
disconnect s = withAxon (Axon.disconnect s)


{- | Replace the *first* matching synapse -}
replaceSynapse
    :: (Monad m, Show s, Eq s)
    => Synapse s -> Synapse s -> Neuron n s -> m (Neuron n s)
replaceSynapse old new = withAxonM (Axon.replaceM old new)


maxDelay :: Neuron n s -> Delay
maxDelay = Axon.maxDelay . axon


withTargets :: (Idx -> Idx) -> Neuron n s -> Neuron n s
withTargets f = withAxon (Axon.withTargets f)


withSynapses :: (s -> s) -> Neuron n s -> Neuron n s
withSynapses f = withAxon (Axon.withSynapses f)


printConnections :: (Show s) => Idx -> Neuron n s -> IO ()
printConnections source n = Axon.printConnections source $ axon n


instance (NFData n, NFData s) => NFData (Neuron n s) where
    rnf (Neuron n ss) = rnf n `seq` rnf ss


instance (Binary n, Binary s) => Binary (Neuron n s) where
    put (Neuron n ss) = put n >> put ss
    get = liftM2 Neuron get get


instance (Show n, Show s) => Show (Neuron n s) where
    showsPrec _ n = shows (ndata n) . showChar '\n' . shows (axon n)



{- To define a network where there's nothing interesting in the neuron itself
 - use Stateless -}
data Stateless = Stateless

instance Binary Stateless where
    put _ = putWord8 0
    get = do
        tag <- getWord8
        case tag of
            0 -> return Stateless
            _ -> fail "Decoding Stateless failed"
