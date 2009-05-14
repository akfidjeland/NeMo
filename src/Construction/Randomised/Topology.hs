module Construction.Randomised.Topology (randomTopology, withProbT) where

import Control.Monad
import Data.List (sort)
import Data.Maybe
import Test.QuickCheck

import Construction.Topology
import Construction.Rand


-- Randomisation for testing
instance (Arbitrary a) => Arbitrary (Topology a) where
    arbitrary = sized genTopology'
        where
            genTopology' 0 = liftM Node arbitrary
            genTopology' n
                    | n > 0 = oneof [liftM Node arbitrary, liftM Cluster (tlist n)]
                    where
                        tlist n = do
                             n' <- choose(1, n)
                             sequence $ replicate n' $ genTopology' (n `div` n')



randomTopology :: Int -> Topology a -> Gen (Topology a)
randomTopology n t
        | n < 1     = error "randomTopology: must choose at least one node"
        | n > sz    = error "randomTopology: too many nodes chosen"
        | otherwise = do
            idx <- uniqueRandomIndices n sz
            -- TODO: deal with errors?
            return $ fromJust $ subtopology (sort idx) t
        where
            sz = sizeT t


-- Return topology where every node is included with probability p
withProbT :: Double -> Topology a -> Gen (Topology a)
withProbT p t = do
        idx <- withProbability p $ indicesT t
        return $ fromJust $ subtopology idx t
