module Examples.Ring where

import NSim
import Construction.Izhikevich

-- Simple network with a ring topology

-- Return neighbour (by index) in cluster of given size
neighbour sz = relativePre f
    where f idx = [(idx +1) `mod` sz]

strongSynapse d = mkSynapse 1000.0 d

exn r = mkNeuron 0.02 0.2 (v + 15*r^2) (8.0-6.0*r^2) v
    where v = -65.0

-- Ring with 'n' neurons and delay of 'd' for each synaspe
ring n d = clusterN (replicate n (fix 0.5 exn))
            [connect every (neighbour n) (strongSynapse d)]
