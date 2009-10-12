{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}

module Test.Construction.Axon (runQC) where

import qualified Data.List as List (sort)
import qualified Data.Map as Map (fromList, fromListWith, assocs)
import qualified Data.Sequence as Seq
import Data.Maybe (isNothing, fromJust)
import Test.QuickCheck

import Construction.Axon
import Construction.Synapse
import Test.Construction.Synapse
import qualified Util.Assocs as Assocs (groupBy, mapElems)
import Types (Source, Target, Delay, Current)


check' :: Testable a => a -> String -> IO ()
check' t str = putStr (str ++ ": ") >> quickCheck t

runQC = do
    check' prop_sort "sort synapses"
    check' prop_sortTime1 "sort axon at different times"
    check' prop_sortTime2 "sort axon at different times (order synapses by delay)"
    check' prop_sortTime3 "sort axon at different times (use fromList for construction)"
    check' prop_connectNonempty "connect non-empty"
    check' prop_connectEmpty "connect empty"
    check' prop_fromList "build axon from list"
    check' prop_connectMany "connect several synapses into axon"
    check' prop_disconnectNonExistent "disconnect non-existent synapse"
    check' prop_disconnect "disconnect synapse"
    check' prop_disconnectOrdering "disconnect synapse"
    check' prop_replaceNonexistent "replace non-existent synapse"
    check' prop_replace "replace existent synapse"
    check' prop_replaceOrdering "replace existent synapse, both ordered and unordered"
    check' prop_withTargets "map function over all target indices"
    check' prop_allPresent "check that all synapses in axon are reported as present"
    check' prop_randomPresent "check random synapses are correctly reported as present"
    check' prop_newPresent "check that new additions are correctly reported as present"
    check' prop_synapsesByDelay "check that returning synapses by delay returns all"
    check' prop_targets "check that targets list is generated correctly"
    check' prop_maxDelay "check that max delay is correctly computed"


type StdSynapse = AxonTerminal Static

{- The synapse collections are sorted by delay, which is present both in the
 - key and the value of the collection. We therefore need to generate our own
 - random collections, to ensure valid data -}
instance Arbitrary (Axon Static) where
    arbitrary = sized $ \nm -> do
        m <- choose (1, 1000)      -- synapses per neuron
        let n = nm `div` m         -- max synapse
        ss <- resize m $ vector nm -- the sources are not used here
        oneof [
            return (Unsorted $ Seq.fromList $ map strip ss),
            return (Sorted ( Map.fromList ( Assocs.mapElems go (Assocs.groupBy delay ss))))
          ]
        where
            go ss = Map.fromListWith (++) $ zip (map target ss) (map (\s -> return (weight s, plastic s, sdata s)) ss)



{- | Sorting a synapse should leave us with the same synapses -}
prop_sort ss = List.sort ss == (List.sort $ terminals $ sort $ Unsorted $ Seq.fromList ss)
    where
        types = ss :: [StdSynapse]


{- | The order of synapses in the axon should be unchanged by the point at
 - which the axon is sorted -}
prop_sortTime1 ss = ss1 == ss2
    where
        ss1 = terminals $ connectMany ss $ sort unconnected
        ss2 = terminals $ sort $ connectMany ss unconnected
        types = ss :: [StdSynapse]


prop_sortTime2 ss = ss1 == ss2
    where
        ss1 = terminalsByDelay $ connectMany ss $ sort unconnected
        ss2 = terminalsByDelay $ sort $ connectMany ss unconnected
        types = ss :: [StdSynapse]


{- | Same, but use fromList to construct unsorted list -}
prop_sortTime3 ss = ss1 == ss2
    where
        ss1 = terminalsByDelay $ connectMany ss $ sort unconnected
        ss2 = terminalsByDelay $ sort $ fromList ss
        types = ss :: [StdSynapse]



{- | After connecting a synapse the synapse collection should be longer by one -}
prop_connectNonempty s axon = size (connect s axon) == 1 + size axon
    where types = (s :: StdSynapse, axon :: Axon Static)


{- | Inserting into an empty axon should give a total of 1 synapse -}
prop_connectEmpty s = size (connect s unconnected) == 1
    where types = (s :: StdSynapse)


{- | Inserting synapses and taking them out again should not change
 - composition, except for the order -}
prop_fromList ss = List.sort ss == List.sort ss_out
    where
        ss_out = terminals $ fromList ss
        types = ss :: [StdSynapse]


{- | Inserting synapses and taking them out again, ordered by delay should not
 - change composition, except for the order -}
prop_synapsesByDelay ss = List.sort ss_out == List.sort ss_out
    where
        ss_out = concatMap strip $ terminalsByDelay $ fromList ss

        strip :: (Delay, [(Target, Current, Bool, Static)]) -> [Current]
        strip (delay, ss) = map (\(_,w,_,_) -> w) ss

        types = ss :: [StdSynapse]


prop_targets ss = List.sort (map target ss) == List.sort (targets (fromList ss))
    where
        types = ss :: [StdSynapse]


prop_maxDelay ss = (not . null) ss ==> maxDelay axon == maximum (map delay ss)
    where
        axon = fromList ss
        types = ss :: [AxonTerminal Static]


{- | Inserting synapses and taking them out again should not change
 - composition, except for the order -}
prop_connectMany ss axon = List.sort (ss ++ terminals axon) == List.sort ss_out
    where
        ss_out = terminals $ connectMany ss axon
        types = (ss :: [StdSynapse], axon :: Axon Static)


{- | Removing a synapse which is not present, should have no effect other than
 - possibly sorting the synapses. -}
prop_disconnectNonExistent s axon =
        s `notElem` ss ==> List.sort ss == List.sort (terminals $ disconnect s axon)
    where
        ss = terminals axon
        types = (s :: StdSynapse, axon :: Axon Static)


{- | Removing a synapse which *is* present, should reduce the synapse count by one -}
prop_disconnect idx axon =
    idx < length ss && idx > 0
        ==> List.sort (s:ss') == List.sort (terminals axon)
    where
        ss = terminals axon
        s = ss !! idx
        ss' = terminals $ disconnect s axon
        types = (idx :: Int, axon :: Axon Static)


{- | Removing a synapse should have the same result, regardless of whether or
 - not the axon is sorted -}
prop_disconnectOrdering idx ss =
        idx < length ss && idx > 0 ==> terminals axon1 == terminals axon2
    where
        s = ss !! idx
        axon1 = modify $ connectMany ss $ sort unconnected -- replace sorted
        axon2 = modify $ fromList ss -- replace unsorted
        modify = either error id . disconnectM s
        types = (idx :: Int, ss :: [StdSynapse])


{- Replacing a non-existent synapse should result in failure -}
prop_replaceNonexistent old new axon =
    old `notElem` (terminals axon) ==> isNothing $ replaceM old new axon
    where
        types = (old, new :: StdSynapse, axon :: Axon Static)



{- Replacing an existing synapse should leave the same number of synapses, but
 - may alter the internal structure -}
-- TODO: read back synapses and make sure delays match
prop_replace idx new axon =
    idx < size axon && idx > 0 ==> size axon == size axon'
    where
        ss = terminals axon
        old = ss !! idx
        axon' = fromJust $ replaceM old new axon
        types = (idx :: Int, new :: StdSynapse, axon :: Axon Static)


{- Replacing a synapse should have the same result regardless of whether or not
 - the axon is already sorted -}
prop_replaceOrdering idx new ss =
        idx < length ss && idx > 0 ==> terminals axon1 == terminals axon2
    where
        old = ss !! idx
        axon1 = replace $ connectMany ss $ sort unconnected -- replace sorted
        axon2 = replace $ fromList ss -- replace unsorted
        replace = either error id . replaceM old new
        types = (idx :: Int, new :: StdSynapse, ss :: [StdSynapse])


{- Check effect of incrementing postsynaptic index for every target -}
prop_withTargets axon = sumTarget ss' == (sumTarget ss + (size axon))
    where
        axon' = withTargets (+1) axon
        ss = terminals axon
        ss' = terminals axon'
        sumTarget = sum . map target
        types = (axon :: Axon Static)


{- Check that a synapse is correctly reported as present -}
prop_allPresent ss = and $ [ present s axon | s <- ss ]
    where
        (Sorted axon) = sort $ fromList ss
        types = ss :: [AxonTerminal Static]


{- Check that random synapses are correctly reported as present -}
prop_randomPresent s ss = present s axon == elem s ss
    where
        (Sorted axon) = sort $ fromList ss
        types = (s :: AxonTerminal Static, ss :: [AxonTerminal Static])


{- Check that when adding a synapse, it's always reported as present -}
prop_newPresent s axon = present s ss
    where
        (Sorted ss) = connect s $ sort axon
        types = (axon :: Axon Static, s :: AxonTerminal Static)
