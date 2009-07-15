-- | A network is constructed topologically, be building a the recursive
-- "Topology" data structure bottom-up. The constructor functions create a
-- complex network from simpler subnets, and then apply one or more connection
-- functions, as defined in "Connectivity" and randomise the network as
-- specified in "Parameterisation".
--
-- As a simple example, the following set of functions create a network with
-- some number of excitatory and inhibitory neurons, all connected.
--
-- @
-- -- excitatory synapse
-- exS = mkRSynapse (between 0.0 0.5) (fixed 1)
-- @
--
-- @
-- -- excitatory neuron
-- exN r = mkNeuron 0.02 0.2 (v + 15*r^2) (8.0-6.0*r^2) v
-- where v = -65.0
-- @
--
-- @
-- --excitatory population
-- exC n  = clusterN (replicate n (randomised exN)) []
-- @
--
-- @
-- -- inhibitory synapse
-- inS = mkRSynapse (between 0.0 (-1.0)) (fixed 1)
-- @
--
-- @
-- -- inhibitory neuron
-- inN r = mkNeuron (0.02 + 0.08*r) (0.25 - 0.05*r) v 2.0 v
-- where v = -65.0
-- @
--
-- @
-- -- inhibitory population
-- inC n = clusterN (replicate n (randomised inN)) []
-- @
--
-- @
-- -- full net
-- net exCount inCount = cluster [exC exCount, inC inCount]
-- [ connect (nth 1) ignorePre exS,
-- connect (nth 2) ignorePre inS ]
-- @
module Construction.Construction (
    cluster,
    clone,
    clusterN,
    build,
    build'
) where

import Test.QuickCheck(generate, Gen)
import System.Random(mkStdGen)

import Types
import Construction.Connectivity
import Construction.Network
import Construction.Neuron
import qualified Construction.Neurons as Neurons (union, fromList)
import Construction.Topology
import Construction.Synapse

{- | Modify all indices in synapses and topology to start from a new base index -}
relocate :: Idx -> Network n s -> Network n s
relocate base = withTerminals (+base)


{- | Return a cluster of subnets transformed according to connector list -}
cluster
    :: (Show n)
    => [Gen (Network n s)]
    -> [Connector n s]
    -> Gen (Network n s)
cluster subnets fs = do
    subnets' <- sequence subnets
    let offsets = scanl (+) 0 (map size subnets')
        relocated = zipWith relocate offsets subnets'
        ns' = Neurons.union $ map networkNeurons relocated
        ts' = map topology relocated
    f $ return $ Network ns' (Cluster ts')
    where
        f = foldr (.) id fs


{- | Return a cluster of subnets each of which is a clone of the others. This
 - should be faster, as we don't need to thread RNG through the whole thing. -}
clone :: (Show n) => Int -> Network n s -> [Connector n s] -> Gen (Network n s)
clone n subnet fs = do
    let subnets = replicate n subnet
        sz = size subnet
        offsets = scanl (+) 0 $ replicate n sz
        -- TODO: share all the following code with cluster
        -- offsets = scanl (+) 0 (map size subnets)
        relocated = zipWith relocate offsets subnets
        ns' = Neurons.union $ map networkNeurons relocated
        ts' = map topology relocated
    f $ return $ Network ns' (Cluster ts')
    where
        f = foldr (.) id fs


{- | Return a cluster of neurons connected according to connector list -}
clusterN :: [Gen n] -> [Connector n s] -> Gen (Network n s)
clusterN ns fs = do
    ns' <- sequence ns
    f $ return $ Network (Neurons.fromList (zip [0..] (map unconnected ns'))) t
    where
        f = foldr (.) id fs
        t = Cluster $ map Node [0..(length ns - 1)]


-- | Generate network by instantiating all random numbers used in network
-- construction.
-- TODO: here we should really use the size of the net to provide a hint to the
-- generate function re how many random numers are required. However, we can't
-- get the size of the net out of the monad without running generate! We could
-- base this on e.g. the number of lines in the description.

build' count seed x = generate count (mkStdGen (fromInteger seed)) x
build = build' 1000
