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


excitatory, inhibitory :: Conductive s => Synapse s -> Bool
excitatory x = (current $! sdata x) > 0
inhibitory x = (current $! sdata x) < 0


data Synapse s = Synapse {
        source :: !Idx,
        target :: !Idx,
        delay  :: !Delay,
        sdata  :: !s  -- variable payload
    } deriving (Eq, Show, Ord)


newtype Static = Static FT deriving (Eq, Show, Ord)


instance Conductive Static where
    current (Static w) = w


instance (NFData s) => NFData (Synapse s) where
    rnf (Synapse s t d pl) = rnf s `seq` rnf t `seq` rnf d `seq` rnf pl `seq` ()


instance NFData Static where
    rnf (Static w) = rnf w `seq` ()


-- TODO: consitent name and argument order
retarget :: Synapse s -> Idx -> Synapse s
retarget (Synapse src tgt d pl) tgt' = Synapse src tgt' d pl


changeSource :: Idx -> Synapse s -> Synapse s
changeSource src' (Synapse src tgt d pl) = Synapse src' tgt d pl


mapIdx :: (Idx -> Idx) -> Synapse s -> Synapse s
mapIdx f (Synapse s t d pl) = Synapse (f s) (f t) d pl
