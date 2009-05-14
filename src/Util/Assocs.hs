{- | Utility functions for association lists. The names conflict with Data.List
 - and other collection types, so it's best to import this qualified. -}

module Util.Assocs where

import qualified Data.List as List
import Data.Function
import Data.Ord(comparing)


keys :: [(a,b)] -> [a]
keys = map fst

elems :: [(a,b)] -> [b]
elems = map snd

mapKeys :: (a1 -> a2) -> [(a1,b)] -> [(a2,b)]
mapKeys f xs = map (\(k,v) -> (f k, v)) xs

mapElems :: (b1 -> b2) -> [(a,b1)] -> [(a,b2)]
mapElems f xs = map (\(k,v) -> (k, f v)) xs

mapAssocs :: (a1 -> a2) -> (b1 -> b2) -> [(a1,b1)] -> [(a2,b2)]
mapAssocs fk fv xs = map (\(k,v) -> (fk k, fv v)) xs

toAssocs :: (a -> k) -> [a] -> [(k,a)]
toAssocs f = map (\x -> (f x, x))

toAssocsWith :: (a -> k) -> (a -> b) -> [a] -> [(k,b)]
toAssocsWith fk fv xs = map (\x -> (fk x, fv x)) xs


{- | Merge two sorted association lists using the provided function to merge
 - equivalent indices. The result is undefined if the two lists are not sorted. -}
mergeBy :: (Ord a) => (b -> b -> b) -> [(a,b)] -> [(a,b)] -> [(a,b)]
mergeBy _ [] ys = ys
mergeBy _ xs [] = xs
mergeBy f (x:xs) (y:ys) = merge x y : mergeBy f xs ys
    where
        merge x@(kx,vx) y@(ky,vy)
            | kx == ky  = (kx, f vx vy)
            | kx < ky   = x
            | otherwise = y

fromGroups :: [(a,[b])] -> [(a,b)]
fromGroups xs = concatMap expand xs
    where expand (x,y) = map ((,) x) y

{- Group according to equality by result of f, return list of associations groups -}
groupBy :: (Ord a) => (b -> a) -> [b] -> [(a, [b])]
groupBy f xs = map (expand f) $ List.groupBy ((==) `on` f) $ List.sortBy (comparing f) xs
    where expand f x = (f (head x), x)


{- | Convert an association list to a densely populated list of a given length,
 - padding missing elements with a default value. -}
densify :: (Integral a) => a -> a -> b -> [(a, b)] -> [b]
densify n m dflt [] = replicate (fromIntegral $ m-n) dflt
densify n m dflt (x:xs)
    | n == m     = []
    | fst x > n  = dflt  : densify (n+1) m dflt (x:xs)
    | fst x == n =
        let h = snd x in
        h `seq` h : densify (n+1) m dflt xs
    | otherwise  = error "densify: out-of-order elements"
