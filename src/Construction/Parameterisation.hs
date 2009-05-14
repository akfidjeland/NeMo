-- | Typically network construction involves an element of randomness. If we
-- want to use random numbers in haskell we need to pass around the random
-- number generator state, which is achieved using the 'Gen' monad. It is
-- desirable to hide this threading of state throughout the program, so we
-- provide some functions to aid with this.
--
-- (Monad-hiding) randomisation comes in two forms:
--
-- 1. Functions where each parameter can be randomised
--
-- 2. Pure functions into which one or more random numbers are injected

module Construction.Parameterisation (
    -- * Randomising parameters
    -- $doc-random-parameters
    between, fixed,
    -- * Randomising pure functions
    -- $doc-randomised-functions
    fix, randomise, randomise2, randomise3, randomised
) where

import System.Random(Random)
import Test.QuickCheck(Gen, choose)

-- $doc-random-parameters
--
-- Some constructors (typically prefixed mkR) accept randomised parameters
-- (of type @Gen a@). These can be created using either 'between' or 'fixed'.
-- For example to define a function which generates synapses with random delay
-- and weight one could use
--
-- @
--  s = mkRSynapse (between 0.0 0.7) (between 1 20)
-- @
--
-- If such a function contains some random and some non-random parameters, the non-random parameters must be specifed using 'fixed'. For example, the same synapse generator with a fixed delay of 10 can be defined as follows
--
-- @
--  s = mkRSynapse (between 0.0 0.7) (fixed 10)
-- @

-- | Return a randomisation parameter in some range
between :: (Random a) => a -> a -> Gen a
between min max = choose (min, max)

-- | Return a randomisation parameter which is fixed to some value
fixed :: a -> Gen a
fixed = return





-- $doc-randomised-functions
-- Randomising a pure function is the preferred approach if a single random
-- number is used for at several points. Consider this function which creates
-- an Izhikevich neuron based on some random number @r@:
--
-- @
-- exN r = mkNeuron 0.02 0.2 (v + 15*r^2) (8.0-6.0*r^2) v
--  where v = -65.0
-- @
--
-- The function @exN@ can be transformed to accept a randomised parameter (and
-- return a result correctly encapsulated in the Gen monad) by using
-- 'randomise', and can thus be used like so:
--
-- @
-- randomise (between 0.0 1.0) exN
-- @
--
-- In fact randomisation in the range [0, 1] is a common case, so a separate
-- function is used, 'randomised'
--
-- @
-- randomised exN
-- @
--
-- Having defined a function like $exN$, one might very well want to use a
-- fixed parameter in some cases which can be achived using either a 'fixed'
-- parameter
--
-- @
-- randomise (fixed 0.5) exN
-- @
--
-- or more elegantly using 'fix'
--
--
-- @
-- fix 0.5 exN
-- @
--
-- Where a higher number of random parameters are required, 'randomise2',
-- randomise3', etc. can be used.


-- | Fix a randomisable function to use a particular parameter
fix :: a -> (a -> b) -> Gen b
fix x f = return $ f x

-- | Randomise function with a randomising paramater
randomise :: Gen a -> (a -> b) -> Gen b
randomise r f = do
    r' <- r
    return $ f r'

-- | Randomise function with a value in the range [0, 1]
randomised :: (Random a, Fractional a) => (a -> b) -> Gen b
randomised = randomise (between 0.0 1.0)


-- | Randomise function with two randomising paramater
randomise2 :: Gen a -> Gen b -> (a -> b -> c) -> Gen c
randomise2 r1 r2  f = do
    r1' <- r1
    r2' <- r2
    return $ f r1' r2'


-- | Randomise function with three randomising parameters
randomise3 :: (Random a, Random b, Random c) =>
    Gen a -> Gen b -> Gen c -> (a -> b -> c -> d) -> Gen d
randomise3 r1 r2 r3 f = do
    r1' <- r1
    r2' <- r2
    r3' <- r3
    return $ f r1' r2' r3'
