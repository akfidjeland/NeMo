{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}


module Construction.Izhikevich where
-- TODO: export list

import Control.Parallel.Strategies (NFData, rnf)
import Control.Monad (liftM)
import Data.Function(on)
import System.Random

import Construction.Neuron
import Construction.Synapse
import Types


{- | Static parameters and initial state for neuron -}
data IzhNeuron f = IzhNeuron {
        paramA :: !f,
        paramB :: !f,
        paramC :: !f,
        paramD :: !f,
        initU :: !f,
        initV :: !f,         -- ^ membrane potential
        -- initial RNG state for per-neuron process
        -- TODO: rename
        stateThalamic :: Maybe (Thalamic f)
    } deriving (Show, Eq)


-- TODO: unbox this
data IzhState f = IzhState {
        stateU :: !f,
        stateV :: !f
    }



stateSigma :: IzhNeuron f -> Maybe f
stateSigma = liftM sigma . stateThalamic


{- | We may want thalamic input, in which case we carry around per-neuron
 - random number generator state, so all neurons can be dealt with in
 - parallel. -}
data Thalamic f = Thalamic {
        sigma :: f,
        rng :: StdGen
    } deriving (Show, Eq)



mkNeuron a b c d v = IzhNeuron a b c d u v Nothing
    where u = b * v

mkNeuron2 a b c d u v s  = IzhNeuron a b c d u v s

-- r is typically a floating point in the range [0,1)
mkThalamic s r = Just $ Thalamic s $ mkStdGen $ round $ r * 100000.0


updateIzh :: Bool -> Current -> IzhState FT -> IzhNeuron FT -> (IzhState FT, Bool)
updateIzh forceFiring i (IzhState u v) (IzhNeuron a b c d _ _ _) =
    if forceFiring || v' >= 30.0
        then (IzhState (u'+d) c, True)
        else (IzhState u' v', False)
    where
        fired v = v >= 30.0
        (u', v') = f $ f $ f $ f (u, v)
        f (u, v) = if fired v then (u, v) else (u', v')
            where
                v' = v + 0.25 * ((0.04*v + 5.0) * v + 140.0 - u + i)
                -- not sure about use of v' or v here!
                u' = u + 0.25 * (a * (b * v' - u))


-- thalamicInput n = maybe (n, 0) (go n) (stateThalamic n)
thalamicInput :: Maybe (Thalamic FT) -> (Maybe (Thalamic FT), Current)
thalamicInput Nothing = (Nothing, 0)
thalamicInput (Just (Thalamic s g0)) = (Just (Thalamic s g2), i')
    where
        (r1, g1) = random g0
        (r2, g2) = random g1
        i' = gauss 0.0 s (r1, r2)




-- Return a random gaussian
-- TODO: move to separate 'random' library
gauss mu sigma (r1, r2) =
    mu + sigma * sqrt (-2 * log r1) * cos (2 * pi * r2)


instance NFData f => NFData (IzhNeuron f) where
    rnf (IzhNeuron a b c d u v s) =
        rnf a `seq`
        rnf b `seq`
        rnf c `seq`
        rnf d `seq`
        rnf u `seq`
        rnf v `seq`
        rnf s


instance (NFData f) => NFData (Thalamic f) where
    rnf (Thalamic s g ) = rnf s `seq` rnf g


instance NFData StdGen where
    rnf s = s `seq` ()


instance Eq StdGen where
    (==) = on (==) show
