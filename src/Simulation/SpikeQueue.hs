{- | For run-time management of signals we need two data structures: a queue of
 - spikes to be delivered in the future, and a read-only (for static networks)
 - data structure containing the relevant synaptic data.
 -
 - The spikes which are generated during simulation are stored in a rotating
 - queue with d entries, where d is the maximum delay in the network. Each
 - entry contains a list (sorted?) of synapses whose spikes should be delivered
 - at that cycle in the future. -}

module Simulation.SpikeQueue (
    SpikeQueue,
    mkSpikeQueue,
    enqSpikes,
    deqSpikes,
    SynapsesRT,
    mkSynapsesRT
) where


import Data.Array.IArray

import Construction.Synapse
import Construction.Network (Network, idxBounds, synapses, maxDelay)
import qualified Simulation.Queue as Q
import qualified Util.Assocs as Assoc (mapElems)
import Types

{- Run-time collection of synapses with fast lookup, grouped by delay so that
 - signals can be quickly inserted into the correct slot in the queue. -}
type SynapsesRT = Array Source [(Delay, [(Target, Weight)])]


mkSynapsesRT :: Network n s -> SynapsesRT
-- TODO: try stripping before getting the assocs
mkSynapsesRT net = array (idxBounds net) $ Assoc.mapElems strip $ synapses net
    where
        strip = Assoc.mapElems (map (\(idx, w, _, _) -> (idx, w)))


{- Rotating queue with insertion at random points, with one slot per delay. -}
type SpikeQueue = Q.Queue (Idx, Current)


{- | Create a new spike queue to handle spikes in the given collection of
 - synapses. -}
mkSpikeQueue :: Network n s -> SpikeQueue
mkSpikeQueue net = Q.mkQueue $! maxDelay net


enqSpikes :: SpikeQueue -> [Idx] -> SynapsesRT -> SpikeQueue
enqSpikes sq fidx ss = foldr Q.enq sq $ concat $ map (ss!) fidx


{- | Dequeue spikes at the head of the queue -}
deqSpikes :: SpikeQueue -> ([(Idx, Current)], SpikeQueue)
deqSpikes sq= (spikes, sq')
    where
        sq' = Q.advance sq
        spikes = Q.head sq'
