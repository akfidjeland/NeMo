module Examples.Random1k (random1k, localClusters', localClusters) where

import System.Random
import Control.Monad
import Control.Parallel.Strategies

import Construction.Izhikevich
import Simulation.FiringStimulus
import NSim hiding (excitatory, random)

import qualified Data.Map as Map
import Construction.Neuron hiding (connect)
import Construction.Network
import Construction.Synapse hiding (excitatory, inhibitory)
import Construction.Topology
import Util.List


-- Simple network as specified in Izhikevich's 2004 paper.

-- excitatory synapse
exS = mkRSynapse (between 0.0 0.5) (fixed 1)

-- excitatory neuron
exN r = mkNeuron2 0.02 b (v + 15*r^2) (8.0-6.0*r^2) u v thalamic
    where
        b = 0.2
        u = b * v
        v = -65.0
        thalamic = mkThalamic 5.0 r

-- excitatory population
exC n  = clusterN (replicate n (randomised exN)) []

-- inhibitory synapse
inS = mkRSynapse (between 0.0 (-1.0)) (fixed 1)

-- inhibitory neuron
inN r = mkNeuron2 (0.02 + 0.08*r) b c 2.0 u v thalamic
    where
        b = 0.25 - 0.05 * r
        c = v
        u = b * v
        v = -65.0
        thalamic = mkThalamic 2.0 r

-- inhibitory population
inC n = clusterN (replicate n (randomised inN)) []

-- full net
random1k exCount inCount = cluster [exC exCount, inC inCount]
    [ connect (nth 1) ignorePre exS,
      connect (nth 2) ignorePre inS ]


-- multiple unconnected clusters of 1024 neurons
-- localRandom1k :: Int -> Gen 
localClusters n = do
    seed <- between 0 100000
    clone n (c seed) []
    where
        c seed = build' 1000000 seed $ random1k 820 204

{-
localClusters' seed c m = Network ns t

    where

        sz = 1024
        n = c * sz
        -- TODO: use different seeds

        -- ns = Map.fromList $ zipWith3 (neuron sz) [0..n-1] (nrands n seed) $ chunksOf m (srands n m seed)
        ns = Map.fromList $ zipNeuron sz m [0..n-1] (nrands n seed) (srands n m seed)

        -- the topology is not used after construction
        t = Node 0

nrands n seed = take n $ randoms $ mkStdGen seed        -- one for each neuron
srands n m seed = take (n * m) $ randoms $ mkStdGen seed  -- one for each synapse

zipNeuron sz m (idx:idxs) (nr:nrs) srs = n : ns
    where
        (sr, srs') = splitAt m srs
        n = neuron sz idx nr sr
        ns = zipNeuron sz m idxs nrs srs'
zipNeuron _ _ _ _ _ = []

-- create either an excitatory or an inhibitory neuron, depending on its index
neuron sz idx nr sr =
    if isExcitatory idx
        then (idx, excitatory sz idx nr sr)
        else (idx, inhibitory sz idx nr sr)
    where
        isExcitatory idx = idx `mod` sz < 820



-- TODO: force evaluation here
exSynapse src base post r  = src `seq` tgt `seq` w `seq` StdSynapse src tgt w 1
    where
        tgt = base + post
        w = 0.5 * r

-- create a single excitatory neuron based
excitatory sz pre nr sr = ss `seq` (exN nr, ss)
    where
        base = pre - (pre `mod` sz)
        ss = zipWith (exSynapse pre base) [0..sz-1] sr
        -- ss = zipWith (\post r -> StdSynapse pre (base+post) (0.5*r) 1) [0..sz-1] sr

-- create a single inhibitory neuron based
inhibitory sz pre nr sr = (inN nr, ss)
    where
        base = pre - (pre `mod` sz)
        ss = zipWith (\post r -> StdSynapse pre (base+post) ((-1.0)*r) 1) [0..sz-1] sr

-}


localClusters' seed c m = Network ns t
    where
        -- random number which is threaded through the whole program
        r = mkStdGen seed
        sz = 1024
        n = c * sz
        ns = Map.fromList $ take n $ rneurons 0 r
        -- ns = Map.fromList $ zipWith3 (neuron sz) [0..n-1] (nrands n seed) $ chunksOf m (srands n m seed)
        -- ns = Map.fromList $ zipNeuron sz m [0..n-1] (nrands n seed) (srands n m seed)

        -- the topology is not used after construction
        t = Node 0

-- Produce an infinite list of neurons
rneurons idx r = (neuron' idx r1) : (rneurons (idx+1) r2)
    where
        (r1, r2) = split r

neuron' idx r =
    if isExcitatory idx
        then (idx, excitatory idx r)
        else (idx, inhibitory idx r)
    where
        -- TODO: remove hard-coding here
        isExcitatory idx = idx `mod` 1024 < 820


excitatory pre r = n `seq` neuron n ss
    where
        (nr, r2) = random r
        n = exN nr
        base = pre - (pre `mod` 1024)
        ss = exSS pre base r2

-- exSS pre base r2 = zipWith (exSynapse pre) [base..base+1023] $ randoms r2
exSS pre base r2 = ss `using` rnf
    where
        ss = map (exSynapse pre) [base..base+1023]

exSynapse src tgt = Synapse src tgt 1 $! Static 0.25
-- exSynapse src tgt r = w `seq` StdSynapse src tgt w 1
    -- where w = 0.5 * r



-- create a single inhibitory neuron based
-- inhibitory pre r = n `seq` (n, ss `using` rnf)
inhibitory pre r = n `seq` neuron n ss
    where
        n = inN nr
        base = pre - (pre `mod` 1024)
        (nr, r2) = random r
        -- ss = zipWith (inSynapse pre) [base..base+1023] $ randoms r2
        ss = map (inSynapse pre) [base..base+1023]


inSynapse src tgt = Synapse src tgt 1 $ Static (-0.5)
-- inSynapse src tgt r = StdSynapse src tgt ((-1.0)*r) 1
