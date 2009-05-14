{-# LANGUAGE MultiParamTypeClasses #-}

module Construction.Neuron (
    NeuronProbe(..),
    mergeProbeFs,
    Neuron,
    ndata,
    unconnected,
    neuron,
    synapses,
    synapsesByDelay,
    connect,
    connectMany,
    disconnect,
    replaceSynapse,
    foldTarget,
    maxDelay,
    withTargets,
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





-- TODO: rename this 'Neuron', and rename typeclass 'Spiking'
-- Make Neuron Izhikevich StdSynapse an instance of Spiking
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


{- | Fold function over target indices -}
foldTarget :: (a -> Idx -> a) -> a -> Neuron n s -> a
foldTarget f x n = Axon.foldTarget f x $! axon n


maxDelay :: Neuron n s -> Delay
maxDelay = Axon.maxDelay . axon


withTargets :: (Idx -> Idx) -> Neuron n s -> Neuron n s
withTargets f = withAxon (Axon.withTargets f)


printConnections :: (Show s) => Idx -> Neuron n s -> IO ()
printConnections source n = Axon.printConnections source $ axon n


instance (NFData n, NFData s) => NFData (Neuron n s) where
    rnf (Neuron n ss) = rnf n `seq` rnf ss


instance (Binary n, Binary s) => Binary (Neuron n s) where
    put (Neuron n ss) = put n >> put ss
    get = liftM2 Neuron get get


instance (Show n) => Show (Neuron n s) where
    show n = show (ndata n) ++ "(" ++ (show $ Axon.size $ axon n) ++ " synapses"
