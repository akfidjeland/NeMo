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
import Construction.Spiking
import Construction.Synapse
import Types


data IzhNeuron f = IzhNeuron {
        paramA :: !f,
        paramB :: !f,
        paramC :: !f,
        paramD :: !f,
        stateU :: !f,
        stateV :: !f,         -- ^ membrane potential
        stateI :: !f,         -- ^ accumulated current
        stateF :: !Bool,
        -- TODO: add strictness?
        stateThalamic :: Maybe (Thalamic f)
    } deriving (Show, Eq)


stateSigma :: IzhNeuron f -> Maybe f
stateSigma = liftM sigma . stateThalamic


{- | We may want thalamic input, in which case we carry around per-neuron
 - random number generator state, so all neurons can be dealt with in
 - parallel. -}
data Thalamic f = Thalamic {
        sigma :: f,
        rng :: StdGen
    } deriving (Show, Eq)

{- | The user can specify parameters to probe, by using a parameter of the the
 - type ProbeFn IzhState. It's tricky to provide a generic function in the user
 - code for probing, since the data structure to use is not known until run-
 - time. The actual implementation of the probing is therefore done using an
 - instance of NeuronProbe -}
data IzhState = U | V | F deriving (Show)


instance (Ord f, Fractional f, Random f, Floating f) => Spiking IzhNeuron f where
    fired = stateF
    update = updateIzh
    addSpike = addSpikeIzh
    preSpikeDelivery n = maybe n (addThalamic n) $ stateThalamic n


mkNeuron a b c d v = IzhNeuron a b c d u v 0 False Nothing
    where u = b * v

mkNeuron2 a b c d u v s  = IzhNeuron a b c d u v 0 False s

-- r is typically a floating point in the range [0,1)
mkThalamic s r = Just $ Thalamic s $ mkStdGen $ round $ r * 100000.0


updateIzh :: (Ord f, Fractional f) => Bool -> IzhNeuron f -> IzhNeuron f
updateIzh forceFiring (IzhNeuron a b c d u v i _ th) =
    if v' >= 30.0 || forceFiring
        then (IzhNeuron a b c d (u'+d) c 0 True th)
        else (IzhNeuron a b c d u' v' 0 False th)
    where
        fired v = v >= 30.0
        (u', v') = f $ f $ f $ f (u, v)
        f (u, v) = if fired v then (u, v) else (u', v')
            where
                v' = v + 0.25 * ((0.04*v + 5.0) * v + 140.0 - u + i)
                -- not sure about use of v' or v here!
                u' = u + 0.25 * (a * (b * v' - u))



addSpikeIzh :: (Num f) => f -> IzhNeuron f -> IzhNeuron f
addSpikeIzh w n = n { stateI = i + w }
    where
        i = stateI n
-- addSpikeIzh w (IzhNeuron a b c d u v i f s) =
--    IzhNeuron a b c d u v (i + w) f s


{-
thalamicInput :: (Floating f, Random f) => IzhNeuron f -> IO (IzhNeuron f)
thalamicInput n =
    maybe
        (return n)
        -- (\s -> return n)
        (\s -> rgauss 0 s >>= \i -> return $ addSpikeIzh i n)
        (stateSigma n)
-}

addThalamic
    :: (Floating f, Random f)
    => IzhNeuron f -> Thalamic f -> IzhNeuron f
addThalamic n th = n { stateI = i + i', stateThalamic = th' }
    where
        (r1, g1) = random $ rng th
        (r2, g2) = random g1
        i' = gauss 0.0 (sigma th) (r1, r2)
        i = stateI n
        th' = Just $ Thalamic (sigma th) g2


-- Return a random gaussian
-- TODO: move to separate 'random' library
gauss mu sigma (r1, r2) =
    mu + sigma * sqrt (-2 * log r1) * cos (2 * pi * r2)


instance NFData f => NFData (IzhNeuron f) where
    rnf (IzhNeuron a b c d u v i f s) =
        rnf a `seq`
        rnf b `seq`
        rnf c `seq`
        rnf d `seq`
        rnf u `seq`
        rnf v `seq`
        rnf i `seq`
        rnf f `seq`
        rnf s


instance (NFData f) => NFData (Thalamic f) where
    rnf (Thalamic s g ) = rnf s `seq` rnf g


instance NFData StdGen where
    rnf s = s `seq` ()


instance Eq StdGen where
    (==) = on (==) show
