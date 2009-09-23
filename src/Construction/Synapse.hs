{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

module Construction.Synapse where
-- TODO: export list


import Control.Parallel.Strategies (NFData, rnf)

import Types


-- TODO: remove this typeclass. It's not needed and the naming is wrong. Also,
-- it's not clear that we'd ever *not* want a weight.
class Conductive s where
    -- TODO: rename weight
    current :: s -> Current


excitatory, inhibitory :: Synapse s -> Bool
excitatory s = (weight s) > 0
inhibitory s = (weight s) < 0


data Synapse s = Synapse {
        source :: {-# UNPACK #-} !Idx,
        target :: {-# UNPACK #-} !Idx,
        delay  :: {-# UNPACK #-} !Delay,
        weight :: {-# UNPACK #-} !Current,
        sdata  :: {-# UNPACK #-} !s  -- variable payload
    } deriving (Eq, Show, Ord)


instance Conductive (Synapse s) where
    current = weight


type Static = ()


instance (NFData s) => NFData (Synapse s) where
    rnf (Synapse s t d w pl) = rnf s `seq` rnf t `seq` rnf d `seq` rnf pl `seq` rnf w `seq` ()


-- TODO: consistent name and argument order
retarget :: Synapse s -> Idx -> Synapse s
retarget (Synapse src tgt d w pl) tgt' = Synapse src tgt' d w pl


changeSource :: Idx -> Synapse s -> Synapse s
changeSource src' (Synapse src tgt d w pl) = Synapse src' tgt d w pl


mapTarget :: (Idx -> Idx) -> Synapse s -> Synapse s
mapTarget f (Synapse s t d w pl) = Synapse s (f t) d w pl


mapIdx :: (Idx -> Idx) -> Synapse s -> Synapse s
mapIdx f (Synapse s t d w pl) = Synapse (f s) (f t) d w pl


mapWeight :: (Current -> Current) -> Synapse s -> Synapse s
mapWeight f (Synapse s t d w pl) = Synapse s t d (f w) pl
