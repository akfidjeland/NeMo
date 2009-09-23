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


import Data.IORef
import Data.Array.IArray

import Construction.Synapse
import Construction.Network (Network, idxBounds, synapses, maxDelay)
import qualified Simulation.Queue as Q
import qualified Util.Assocs as Assoc (mapElems)
import Types

{- Run-time collection of synapses with fast lookup, grouped by delay so that
 - signals can be quickly inserted into the correct slot in the queue. -}
type SynapsesRT = Array Idx [(Delay, [(Idx, Current)])]


mkSynapsesRT :: Network n s -> SynapsesRT
mkSynapsesRT net = array (idxBounds net) $ Assoc.mapElems strip $ synapses net
    where
        strip = Assoc.mapElems (map (\(idx, w, _) -> (idx, w)))


{- Rotating queue with insertion at random points, with one slot per delay. -}
-- TODO: is IORef really required?
type SpikeQueue = IORef (Q.Queue (Idx, Current))


{- | Create a new spike queue to handle spikes in the given collection of
 - synapses. -}
mkSpikeQueue :: Network n s -> IO (SpikeQueue)
mkSpikeQueue net = newIORef $! Q.mkQueue $! maxDelay net


enqSpikes :: SpikeQueue -> [Idx] -> SynapsesRT -> IO ()
enqSpikes sq fidx ss = modifyIORef sq (aux fidx ss)
    where aux fidx ss sq = foldr Q.enq sq $ concat $ map (ss!) fidx


{- | Dequeue spikes at the head of the queue -}
deqSpikes :: SpikeQueue -> IO [(Idx, Current)]
deqSpikes sq = do
    advanceSpikeQueue sq
    sq' <- readIORef sq
    return $! Q.head sq'


advanceSpikeQueue sq = modifyIORef sq Q.advance
