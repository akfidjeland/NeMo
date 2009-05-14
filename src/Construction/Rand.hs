{- | Various randomisation function useful when building networks -}
module Construction.Rand (
    randomList,
    randomSublist,
    withProbability,
    uniqueRandomIndices
) where

import System.Random
import Test.QuickCheck
import List (sort)


-------------------------------------------------------------------------------
-- Random numbers
-------------------------------------------------------------------------------


-- Return a lazy list of random numbers in the range [0, m]
randomList :: (Num a, Random a) => a -> Gen [a]
randomList m = sequence $ repeat $ choose (0, m)


uniqueRandomList :: (Num a, Random a) => a -> Int -> Int -> [a] -> Gen [a]
uniqueRandomList m nMax n used
    | nMax < 0  = error "uniqueRandomList: cannot take negative number of values"
    | n == nMax = return []
    | otherwise = do
        r <- freshRandom m used
        rs <- uniqueRandomList m nMax (n+1) (r:used)
        return (r:rs)


freshRandom :: (Num a, Random a) => a -> [a] -> Gen a
freshRandom m acc = do
    rs <- randomList m
    return $ head $ filter (`notElem` acc) rs


-- Return a list of n unique indices for a list of length m
uniqueRandomIndices :: Int -> Int -> Gen [Int]
uniqueRandomIndices n m
    | m == n    = return [0..(m-1)] -- we probably want to deal separately
                                    -- with the case where we select most
                                    -- of the available indices
    | n > m     = error "uniqueRandomIndices: cannot generate enough indices"
    | otherwise = uniqueRandomList (m-1) n 0 []


-- Return n random entries in a list
randomSublist :: Int -> [a] -> Gen [a]
randomSublist n xs' = do
    idx <- uniqueRandomIndices n (length xs')
    return $ randomSublist' (sort idx) 0 xs'
    where
        randomSublist' [] _ _ = []
        randomSublist' _ _ [] = error $ "randomSublist: indices out of range"
        randomSublist' (i:is) n (x:xs)
            | n == i    = x : randomSublist' is (n+1) xs
            | otherwise = randomSublist' (i:is) (n+1) xs


-- Return a single random list entry
randomElem :: [a] -> Gen a
randomElem xs = do
        i <- choose(0, length xs - 1)
        return $ xs !! i


-- Return a sublist where each element of the original is included with
-- probability p
withProbability :: (Floating f, Random f, Ord f) => f -> [a] -> Gen [a]
withProbability p xs = do
        rs <- randomList 1.0
        return $ map fst $ filter (\rx -> (snd rx) < p) $ zip xs rs
