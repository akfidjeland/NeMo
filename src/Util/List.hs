module Util.List (
    chunksOf,
    maxOr0,
    groupUnsortedBy,
    groupUnsorted,
    replace,
    replaceBy
) where

import Data.Function (on)
import Data.Ord (comparing)
import Data.List (sortBy, groupBy)

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = y : chunksOf n ys
    where (y, ys) = splitAt n xs


maxOr0 [] = 0
maxOr0 xs = maximum xs


-- | Group a potentially unsorted list
groupUnsortedBy :: (Eq b, Ord b) => (a -> b) -> [a] -> [[a]]
groupUnsortedBy f xs = groupBy ((==) `on` f) $ sortBy (comparing f) xs

groupUnsorted :: (Eq a, Ord a) => [a] -> [[a]]
groupUnsorted = groupUnsortedBy id


replaceBy :: (a -> a -> Bool) -> a -> a -> [a] -> [a]
replaceBy _  _   _   []     = []
replaceBy eq old new (y:ys) =
    if y `eq` old
        then new : ys
        else y : replaceBy eq old new ys

-- | Replace the first matching (by equality) list element
replace :: (Eq a) => a -> a -> [a] -> [a]
replace = replaceBy (==)
