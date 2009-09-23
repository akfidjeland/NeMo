{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

{- | Connectivity within a network is specified using a number of connector
 - functions, which in turn makes use of selector functions. -}
module Construction.Connectivity (
    -- * Connectors
    Connector,
    connect, reconnect,
    -- * Selectors
    -- $doc-selectors
    SourceSelector, TargetSelector,

    -- ** Selector chaining
    -- $doc-selector-chaining
    (|>),

    -- ** Pruning selector functions
    Selector,
    every, oneof, random, withProb, only, excluding, nth,

    -- ** Relative selectors
    -- | Selection of a post-synaptic neuron can be based on the pre-synaptic
    -- neuron. A post-synaptic selection rule should thus start with a relative
    -- selector.
    nonself, cca, ccaRoot, relativePre, ignorePre,

    -- ** Synapse-predicate selectors
    -- $doc-synapse-predicate
    everySynapse, anySynapse,

    -- ** Synapse selectors
    -- $doc-synapse-selectors
    synapses,
    presynaptic,
    postsynaptic
) where

import Control.Monad
import Data.Maybe
import Test.QuickCheck (Gen, choose)

import Construction.Network hiding (synapses, synapsesOf)
import Construction.Neuron hiding (connect, synapses)
import Construction.Neurons (Neurons, updateSynapses, addSynapseAssocs, synapsesOf)
import Construction.Synapse
import Construction.Rand
import Construction.Topology
import Construction.Randomised.Topology
import Types



-- $doc-selectors
-- Connections are defined using selectors which map a topology to a list of
-- indices. A connection is defined using a selector for the presynaptic neuron
-- ('SourceSelector') and one selector ('TargetSelector') for the postsynaptic
-- neuron. Connections are set up by taking each neuron index returned by the
-- source selector and connecting it to every index returned by the target.
-- Composite selectors can be built up by combining a number of selection
-- primitives.

type SourceSelector n s a =     (Neurons n s, Topology a)  -> Gen (Neurons n s, Topology a)
type TargetSelector n s a = (a, (Neurons n s, Topology a)) -> Gen (Neurons n s, Topology a)




-- $doc-selector-chaining Selectors can be chained in various ways using the '|>' function.
-- E.g. to select 10 random post-synaptic neurons, which should differ from
-- the pre-synaptic neuron use
--
-- @
--      (nonself |> random 10)
-- @

-- | Chain two selector functions. (This is left-to-right Kleisli composition
-- of monads normally defined in Control.Monad as (>=>).)
(|>)   :: Monad m => (a -> m b) -> (b -> m c) -> (a -> m c)
f |> g = \x -> f x >>= g




--  returning a smaller collection. We can prune a collection c containing (possibly among other things) data of
-- type e, using the following functions. Instances should obey the following:
--  every              == return
--  size (oneof xs)    == 1
--  size (random n xs) == n
--  TODO: perhaps remove the Neurons, and instead make Network n an instance

-- | Selector functions are used to select a subset of a collection (of
-- neurons, synapses, etc). The collection of neurons is passed through
-- selector functions, so that the selectors can query properties of neurons
-- and synapses.
class Selector c e n s where
    -- | Return the collection unchanged
    every     ::                (Neurons n s, c e) -> Gen (Neurons n s, c e)
    -- | Return the collection with only a single random element remaining
    oneof     ::                (Neurons n s, c e) -> Gen (Neurons n s, c e)
    -- | Return the collection with n random elements remaining
    random    :: Int         -> (Neurons n s, c e) -> Gen (Neurons n s, c e)
    -- | Return the collection with each element retained with probability p
    withProb  :: Double      -> (Neurons n s, c e) -> Gen (Neurons n s, c e)
    -- | Return the collection with only the elements for which the predicate is true
    only      :: (e -> Bool) -> (Neurons n s, c e) -> Gen (Neurons n s, c e)
    -- | Return the collection with only the elements for which the predicate is not true
    excluding :: (e -> Bool) -> (Neurons n s, c e) -> Gen (Neurons n s, c e)
    -- | Return only the nth element, counting from 1 (e.g. list element,
    -- sub-topology etc)
    nth       :: Int         -> (Neurons n s, c e) -> Gen (Neurons n s, c e)
    -- Return only the first element (e.g. list element, sub-topology etc)
    -- fstP      ::                (Neurons n, c e) -> Gen (Neurons n, c e)
    -- Return only the second element (e.g. list element, sub-topology etc)
    -- sndP      ::                (Neurons n, c e) -> Gen (Neurons n, c e)

    -- | Convert collection to a list. This is generally not required in
    -- writing connetion rules, but is used by the connector functions.
    flatten   ::                (Neurons n s, c e) -> Gen [e]
    -- defaults:
    every          = return
    excluding p xs = only (not . p) xs
    -- fstP           = nth 0
    -- sndP           = nth 1


instance Selector [] a n s where
    oneof       = liftSN oneof'
        where oneof' xs = do { idx <- choose (0, length xs-1); return [xs !! idx] }
    random n    = liftSN $ randomSublist n
    withProb p  = liftSN $ withProbability p
    only p      = liftSNM $ filter p
    nth n       = liftSNM $ (:[]) . (!!(n-1))
    -- fstP        = liftSNM $ (:[]) . head
    -- sndP        = liftSNM $ (:[]) . head . tail
    flatten     = return . snd


instance Selector Topology a n s where
    oneof       = liftSN $ randomTopology 1
    random n    = liftSN $ randomTopology n
    withProb p  = liftSN $ withProbT p
    -- TODO: how do we deal with an empty Topology?
    only p      = liftSNM $ only' p
        where only' p xs = fromJust $ filterT p xs
    nth n      = liftSNM $ nthT (n-1)
    flatten     = return . flattenT . snd




liftSN :: (Selector c e n s) =>
        (c e -> Gen (c e)) -> (Neurons n s, c e) -> Gen (Neurons n s, c e)
liftSN f (net, xs) = do
        xs' <- f xs
        return (net, xs')


liftSNM :: (Selector c e n s) => (c e -> c e) -> (Neurons n s, c e) -> Gen (Neurons n s, c e)
liftSNM f (net, xs) = return (net, f xs)



-- Return sub-topology rooted at nth parent of selector node, but only the part
-- which does *not* contain the selector node itself
unrelatedR :: (Eq a) => Int -> a -> Topology a -> Topology a
unrelatedR n s t = excludesR s (sharedAncestorR n s t)


-- | Return topology with closest common ancestor (cca) n levels above x.
cca n = liftRSel (unrelatedR n)


-- | Return topology where the root node is the closest common ancestor (cca)
-- in the topology.
ccaRoot :: (Idx, (Neurons n s, Topology Idx)) -> Gen (Neurons n s, Topology Idx)
ccaRoot = liftRSel excludesP

-- | Ignore the pre-synaptic neuron
ignorePre = liftRSel (\x t -> t)


-- Return sub-topology rooted at nth parent of selector node, but ignoring the
-- part which does not contain the selector node.
relatedR :: (Eq a) => Int -> (a, Topology a) -> Topology a
relatedR n (s, t) = includesR s (sharedAncestorR n s t)


-- Pruning selectors

liftRSel :: (b -> Topology a -> Topology a) ->
        (b, (Neurons n s, Topology a)) -> Gen (Neurons n s, Topology a)
liftRSel f (s, (n, t)) = return $ (n, f s t)

-- Return the whole topology excluding the selector
-- TODO: base this on indices rather than equality
nonselfP :: (Eq a) => a -> Topology a -> Topology a
nonselfP s t = fromJust $ filterT (/=s) t

-- | Return the topology with excluding the presynaptic neuron
nonself :: (Eq a) => (a, (Neurons n s, Topology a)) -> Gen (Neurons n s, Topology a)
nonself = liftRSel nonselfP


self :: (a, Topology a) -> [a]
self (pre, _) = [pre]


-- | Return the topology containing only the neurons whose indices is a
-- function of the index of the presynaptic neuron.  For example, to select a
-- neighbour (by index) in a neuron population of some given size one can use:
--
-- @
-- neighbour sz = relativePre f
--  where f idx = [(idx +1) `mod` sz]
-- @
relativePre :: (Eq a) =>
    (a -> [a]) -> (a, (Neurons n s, Topology a)) -> Gen (Neurons n s, Topology a)
relativePre f (pre, net) = only (`elem` (f pre)) net





-------------------------------------------------------------------------------
-- Synapse-predicate selectors

-- $doc-synapse-predicate
-- For selection of neurons based on synapse properties

{- | Return sub-topology containing only neurons for which the predicate is
 - true for every synapse -}
everySynapse
    :: (Synapse s -> Bool)
    -> (Neurons n s, Topology Idx)
    -> Gen (Neurons n s, Topology Idx)
everySynapse p (ns, t) = return $ (ns, fromJust (filterT f t))
    where
        f idx = all p (synapsesOf ns idx)


{- | Return sub-topology containing only neurons for which the predicate is
 - true for every synapse -}
anySynapse
    :: (Synapse s -> Bool)
    -> (Neurons n s, Topology Idx)
    -> Gen (Neurons n s, Topology Idx)
anySynapse p (ns, t) = return $ (ns, fromJust (filterT f t))
    where
        f idx = any p (synapsesOf ns idx)


-------------------------------------------------------------------------------
-- Synapse selectors

-- $doc-synapse-selectors
-- Selection may also take place at the synapse level rather than at the neuron
-- level, e.g. to 'reconnect' or 'disconnect' existing synapses. A synapse
-- selector starts with a Topology and should end with a list of synapses. It
-- is often useful to just operate on a list synapses at some stage of the
-- selection rule. The pruning selectors are defined both over topologies and
-- lists, so the same selectors can be used.
--
-- As an example, to select a single random excitatory synapse use
--
-- @
--      synapses |> only excitatory |> oneof
-- @

{- | Return all synapses of all neurons in topology -}
synapses :: (Neurons n s, Topology Idx) -> Gen (Neurons n s, [Synapse s])
synapses (ns, t) = return $ (ns, concatMap (synapsesOf ns) (flattenT t))


-- TODO: use this type throughout
-- type RelativeSel a = (a, Neurons n, Topology Idx) -> Gen (Neurons n, Topology Idx)

{- | Switch from relative synapse-based selection to relative neuron-based
 - selection using the presynaptic neuron. -}
presynaptic
    :: (Synapse s, (Neurons n s, Topology Idx))
    -> Gen (Idx, (Neurons n s, Topology Idx))
presynaptic (s, (ns, t)) = return $! (source s, (ns, t))


{- | Switch from relative synapse-based selection to relative neuron-based
 - selection using the postsynaptic neuron. -}
postsynaptic
    :: (Synapse s, (Neurons n s, Topology Idx))
    -> Gen (Idx, (Neurons n s, Topology Idx))
postsynaptic (s, (ns, t)) = return $! (target s, (ns, t))



-------------------------------------------------------------------------------
-- Connections


-- | Connectors are functions which transform a network, normally by
-- manipulating the synapse lists associated with each neuron.
type Connector n s = Gen (Network n s) -> Gen (Network n s)

type SGen s = Idx -> Idx -> Gen (Synapse s)

{- | Add a set of connections according to connection rules -}
connect
    :: SourceSelector n s Idx
    -> TargetSelector n s Idx
    -> SGen s
    -> Gen (Network n s)
    -> Gen (Network n s)
connect fPre fPost gSyn net = do
    (Network ns t) <- net
    pres <- (fPre |> flatten) (ns, t)
    posts <- sequence $ map ((flip (curry fPost)) (ns, t) |> flatten) pres
    ss' <- sequence $ map (sequence . connect') $ zip pres posts
    -- TODO: maintain ordering of synapses
    -- TODO: use withNeurons here use
    return $ Network (addSynapseAssocs (zip pres ss') ns) t
    where
        connect' (x, ys) = [ gSyn x y | y <- ys ]


type SynapseS n s = (Neurons n s, Topology Idx) -> Gen (Neurons n s, [Synapse s])
-- TODO: NeuronS need not be based on topology only, but should instead be based on pre and net
-- TODO: enforce termination in a *single* neuron
type NeuronS n s = (Synapse s, (Neurons n s, Topology Idx)) -> Gen (Neurons n s, Topology Idx)



{- Reconnect existing synapses, keeping the existing synapse properties.
 - Selection should return a list of synapses. Selection may be based on
 - neurons or on synapses. Selection starts with the whole network, and can
 - proceed by limiting the set of candidate neurons, but must terminate with
 - returning the list of synapses, which can be further reduced. If
 - reconnection is done based on synapse properties on its own, selection can
 - start with a reduction of the network to the full list of synapses in the
 - net. -}
reconnect
    :: (Eq s, Show s)
    => SynapseS n s
    -> NeuronS n s
    -> Gen (Network n s)
    -> Gen (Network n s)
reconnect preS postS mnet = do
    net@(Network ns t) <- mnet
    old <- (preS |> flatten) (ns, t)
    newIdx <- mapM (((flip (curry postS)) (ns, t)) |> flatten) old
    -- newIdx should be a singleton list
    let new = zipWith retarget (map head newIdx) old
    let repl = zip old new
    return $! withNeurons (updateSynapses repl) net

-- TODO: add tests to verify that the length of synapse list does not change!
