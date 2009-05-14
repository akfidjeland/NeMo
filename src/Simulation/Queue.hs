{- Queue of lists with insertion at random points, implemented as a rotating
 - buffer. -}

module Simulation.Queue (
    Queue,
    mkQueue,
    head,
    enq,
    advance
) where

import Array
import Prelude hiding(head)


data Queue a = Queue {
        first :: Int,          -- index of the front of the queue, i.e. delay=1
        queue :: Array Int [a]
    } deriving Show


mkQueue :: Int -> Queue a
mkQueue len = Queue 0 (array (0, len) assoc)
    where
        assoc = [(i, []) | i <- [0..len] ]


head :: Queue a -> [a]
head (Queue h q) = q!h


-- Increment time by rotating the buffer and clearing last entry
advance :: Queue a -> Queue a
advance sq = Queue next (queue sq // [(last, [])])
    where
        maxOffset = rangeSize $ bounds $ queue sq
        offset d = (first sq + d) `mod` maxOffset
        next = offset 1
        last = offset maxOffset


enq :: (Int, [a]) -> Queue a -> Queue a
enq (delay, spikes) (Queue h q) = Queue h (q // diff)
    where
        maxOffset = rangeSize $ bounds $ q
        offset = (h + delay) `mod` maxOffset
        diff = [(offset, (spikes ++ q!offset))]

-- prop: adding a list of synapses with delay 0, the list should be a sublist of head
