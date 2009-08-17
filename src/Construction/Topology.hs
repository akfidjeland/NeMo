module Construction.Topology where

import Control.Monad
import Control.Parallel.Strategies (NFData, rnf)
import Data.Binary
import Data.Maybe
import qualified Data.List as List


data Topology a
        = Node a
        | Cluster [Topology a]
        | NoTopology
    deriving (Show, Eq)


instance Functor Topology where
    fmap f (Node x)     = Node (f x)
    fmap f (Cluster xs) = Cluster $ map (fmap f) xs


-- Serialisation
instance (Binary a) => Binary (Topology a) where
    put (Node x) = putWord8 0 >> put x
    put (Cluster xs) = putWord8 1 >> put xs
    get = do
        tag <- getWord8
        case tag of
            0 -> liftM Node get
            1 -> liftM Cluster get
            _ -> error "Could not deserialise Topology"


instance NFData a => NFData (Topology a) where
    rnf (Node x) = rnf x
    rnf (Cluster xs) = rnf xs


-- functions on Topology instances fall into several classes. First, some
-- return a list of neurons. These functions are postfixed 'N'. Second, some
-- functions return a tree pruned at the nodes, and thus do not alther the
-- height of the topology. These are postfixed 'B'. Third, some functions
-- reduce the data structure from its root in a top-down manner,
-- by either returning a sub-topology or returning the same topology with an
-- entire sub-topology removed.

-- TODO: use F=flattening, P=pruning, R=reducing


-- Return number of nodes in topology
sizeT :: Topology a -> Int
sizeT (Node _)     = 1
sizeT (Cluster xs) = foldl (\a t -> a + sizeT t) 0 xs


-- Return the list of indices of the nodes in the topology
-- | TODO: can we guarantee indices in this range?
indicesT :: Topology a -> [Int]
indicesT t = [0..sizeT t-1]


-- Return list of nodes
flattenT :: Topology a -> [a]
flattenT (Node x) = [x]
flattenT (Cluster xs) = concat $ map flattenT xs


zipT :: Topology a -> Topology b -> Topology (a, b)
zipT (Node x) (Node y) = Node (x, y)
zipT (Cluster xs) (Cluster ys) = Cluster $ zipWith zipT xs ys
zipT _ _ = error "zipT: topologies are not isomorphic"


unzipT :: Topology (a, b) -> (Topology a, Topology b)
unzipT (Node (x, y)) = (Node x, Node y)
unzipT (Cluster zs)  = (Cluster xs, Cluster ys)
    where
        (xs, ys) = unzip $ map unzipT zs


zipWithT f xs ys = fmap (uncurry f) $ zipT xs ys


-- Return a topology containing only nodes satisfying a predicate
filterT :: (a -> Bool) -> Topology a -> Maybe (Topology a)
filterT p (Node x)
        | p x       = Just $ Node x
        | otherwise = Nothing
filterT p (Cluster xs)
        | null xs'  = Nothing
        | otherwise = Just $ Cluster xs'
        where
            xs' = mapMaybe (filterT p) xs


-- Return an isomorphic topology with values replaced by indices
number t = mapAccumL f 0 t
    where f acc _ = (acc+1, acc)


-- Return only part of the topology containing only the nodes at the given
-- indices (as assigned by 'number').
-- pre: sorted idx
--      forall x in idx . 0 <= x < size topology
-- TODO: issue warning if idx contains invalid entries?
subtopology :: [Int] -> Topology a -> Maybe (Topology a)
subtopology idx t = (liftM (fmap snd)) $ filterT fst $ snd $ mapAccumL aux (0, idx) t
    where
        aux (acc, []) x  = ((acc+1, []), (False, x))
        aux (acc, i:is) x
            | i == acc   = ((acc+1, is), (True, x))
            | otherwise  = ((acc+1, (i:is)), (False, x))


nthT :: Int -> Topology a -> Topology a
-- nthT n t = fromJust $ subtopology [n] t
nthT n (Node x)     = error "nthT: cannot reduce node"
nthT n (Cluster xs) = Cluster [xs!!n]


mapAccumL :: (acc -> x -> (acc, y)) -> acc -> Topology x -> (acc, Topology y)
mapAccumL f acc (Node x)     = (acc', Node x')
    where (acc', x') = f acc x
mapAccumL f acc (Cluster xs) = (acc', Cluster xs')
    where (acc', xs') = List.mapAccumL (mapAccumL f) acc xs



-- Returns sub-topology containing the selector
includesR :: (Eq a) => a -> Topology a -> Topology a
includesR s (Node x)
        | s == x    = Node x
        | otherwise = undefined
includesR s (Cluster xs) = head $ filter (s `elemT`) xs


-- Returns sub-topology which does not contain the selector
excludesR :: (Eq a) => a -> Topology a -> Topology a
excludesR = excludesP


-- Returns topology pruned of all sub-topologies which do not contain the selector.
includesP :: (Eq a) => a -> Topology a -> Topology a
includesP s (Node x)
        | x == s    = Node x
        | otherwise = undefined
includesP s (Cluster xs)
        | null xs'  = undefined
        | otherwise = Cluster xs'
        where
            xs' = filter (elemT s) xs


-- Returns topology pruned of all sub-topologies which are not members of the
-- selector.
-- TODO: improve naming scheme here!
-- TODO: do this only based on indices, to avoid Eq requirement
-- TODO: remove this whole function
includesP' :: (Eq a) => [a] -> Topology a -> Topology a
includesP' s (Node x)
        | x `elem` s = Node x
        | otherwise  = undefined
includesP' s (Cluster xs)
        | null xs'   = undefined
        | otherwise  = Cluster xs'
        where
            xs' = filter (containsT s) xs



-- Returns topology pruned of the sub-topology which contains the selector
excludesP :: (Eq a) => a -> Topology a -> Topology a
excludesP s (Node x)
        | s == x    = error "Topology.excludesP: cannot exclude singleton node"
        | otherwise = Node x
excludesP s (Cluster xs)
        | null xs'  = error $ "Topology.excludesP: empty cluster list"
        | otherwise = Cluster xs'
        where
            xs' = filter (not . elemT s) xs


-- Return sub-topology rooted at nth parent of selector node
-- TODO: find a better name
sharedAncestorR :: (Eq a) => Int -> a -> Topology a -> Topology a
sharedAncestorR n s t
        | s `elemT` t && n == depthOf s t = t
        | s `elemT` t && n < depthOf s t  = sharedAncestorR n s (includesR s t)
        | otherwise                       = undefined


depthOf s t = depth $ includesP s t


-- Return max depth of topology
depth :: Topology a -> Int
depth (Node _) = 0
depth (Cluster xs) = 1 + maximum (map depth xs)


-- TODO: use more sensible data structure in the leaf nodes of Topology to speed this up.
elemT :: (Eq a) => a -> Topology a -> Bool
elemT s t = s `elem` flattenT t


-- Return true if topology contains at least one member of selector
containsT :: (Eq a) => [a] -> Topology a -> Bool
containsT s t = any (`elem` (flattenT t)) s
