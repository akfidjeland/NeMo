{- | This module gathers firing statistics at run-time -}

module Simulation.Statistics (Statistics, newStatistics, update, firingRate) where

import Data.Array.ST
import Data.Array.Unboxed

import Types (Idx, Time, FiringOutput(..))

data Statistics = Statistics {
        cycles :: Time,
        counts :: FiringCount
    }


type FiringCount = UArray Idx Int


newStatistics :: (Idx, Idx) -> Statistics
newStatistics bounds = Statistics 0 $ listArray bounds $ repeat 0


{- | Increment per-neuron firing counters for neurons that just fired -}
updateCounts :: [Idx] -> FiringCount -> FiringCount
updateCounts ns iarr = do
    -- TODO: use unsafe thaw to avoid copying here
    runSTUArray $ do -- modify array in place
        marr <- thaw iarr
        mapM_ (inc marr) ns
        return marr
    where
        inc arr n = writeArray arr n . (+1) =<< readArray arr n


update :: FiringOutput -> Statistics -> Statistics
update (FiringOutput xs) stats = Statistics cycles' counts'
    where
        cycles' = cycles stats + 1
        counts' = updateCounts xs $ counts stats


{- | Return average firing rate for the whole network -}
firingRate :: Statistics -> Double
firingRate stats = total / (realToFrac $ cycles stats)
    where
        total = realToFrac $ sum $ elems $ counts stats
