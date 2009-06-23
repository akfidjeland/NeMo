module Examples.Random1k (random1k) where

import Construction.Izhikevich
import Simulation.FiringStimulus
import NSim hiding (excitatory, random)

import Construction.Neuron hiding (connect)
import Construction.Network
import Construction.Synapse hiding (excitatory, inhibitory)
import Construction.Topology


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
