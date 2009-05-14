{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}

module Test.Construction.Axon (runQC) where

import Data.List (sort)
import qualified Data.Map as Map (fromList, assocs)
import Data.Maybe (isNothing, fromJust)
import Test.QuickCheck

import Construction.Axon
import Construction.Synapse
import Test.Construction.Synapse
import qualified Util.Assocs as Assocs (groupBy, mapElems)


check' :: Testable a => a -> String -> IO ()
check' t str = putStr (str ++ ": ") >> quickCheck t

runQC = do
    check' prop_connectNonempty "connect non-empty"
    check' prop_connectEmpty "connect empty"
    check' prop_fromList "build axon from list"
    check' prop_connectMany "connect several synapses into axon"
    check' prop_disconnectNonExistent "disconnect non-existent synapse"
    check' prop_disconnect "disconnect synapse"
    check' prop_replaceNonexistent "replace non-existent synapse"
    check' prop_replace "replace existent synapse"
    check' prop_withTargets "map function over all target indices"


type StdSynapse = Synapse Static

{- The synapse collections are sorted by delay, which is present both in the
 - key and the value of the collection. We therefore need to generate our own
 - random collections, to ensure valid data -}
instance Arbitrary (Axon Static) where
    arbitrary = sized $ \nm -> do
        m <- choose (1, 1000) -- synapses per neuron
        let n = nm `div` m    -- max synapse
        ss <- resize m $ vector nm
        return $ Axon $ Map.fromList $ Assocs.mapElems (map strip) $ Assocs.groupBy delay ss


{- | After connecting a synapse the synapse collection should be longer by one -}
prop_connectNonempty s axon = size (connect s axon) == 1 + size axon
    where types = (s :: StdSynapse, axon :: Axon Static)


{- | Inserting into an empty axon should give a total of 1 synapse -}
prop_connectEmpty s = size (connect s unconnected) == 1
    where types = (s :: StdSynapse)


{- | Inserting synapses and taking them out again should not change
 - composition, except for the order -}
prop_fromList ss axon = sort ss_in == sort ss_out
    where
        source = 0
        -- all synapses in the same axons have same source
        ss_in = map (changeSource source) ss
        ss_out = synapses source $ fromList ss_in
        types = (ss :: [StdSynapse], axon :: Axon Static)


{- | Inserting synapses and taking them out again should not change
 - composition, except for the order -}
prop_connectMany ss axon = sort (ss_in ++ synapses source axon) == sort ss_out
    where
        source = 0
        ss_in = map (changeSource source) ss
        ss_out = synapses source $ connectMany ss_in axon
        types = (ss :: [StdSynapse], axon :: Axon Static)


{- | Removing a synapse which is not present, should have no effect -}
prop_disconnectNonExistent s axon =
        s `notElem` ss ==> ss == (synapses source $ disconnect s axon)
    where
        source = 0
        ss = synapses source axon
        types = (s :: StdSynapse, axon :: Axon Static)


{- | Removing a synapse which *is* present, should reduce the synapse count by one -}
prop_disconnect idx axon =
    idx < length ss && idx > 0
        ==> sort (s:ss') == sort (synapses source axon)
    where
        source = 0
        ss = synapses source axon
        s = ss !! idx
        ss' = synapses source $ disconnect s axon
        types = (idx :: Int, axon :: Axon Static)


{- Replacing a non-existent synapse should result in failure -}
prop_replaceNonexistent old new axon =
    old `notElem` (synapses 0 axon)
        ==> isNothing $ replaceM old new axon
    where
        types = (old, new :: StdSynapse, axon :: Axon Static)



{- Replacing an existing synapse should leave the same number of synapses, but
 - may alter the internal structure -}
-- TODO: read back synapses and make sure delays match
prop_replace idx new axon =
    idx < size axon && idx > 0 ==> size axon == size axon'
    where
        ss = synapses 0 axon
        old = ss !! idx
        axon' = fromJust $ replaceM old new axon
        types = (idx :: Int, new :: StdSynapse, axon :: Axon Static)


{- Check effect of incrementing postsynaptic index for every target -}
prop_withTargets axon = sumTarget ss' == sumTarget ss + (size axon)
    where
        axon' = withTargets (+1) axon
        ss = synapses 0 axon
        ss' = synapses 0 axon'
        sumTarget = sum . map target
        types = (axon :: Axon Static)
