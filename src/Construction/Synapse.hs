{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

module Construction.Synapse where
-- TODO: export list

import Control.Monad (liftM, liftM4)
import Control.Parallel.Strategies (NFData, rnf)
import Data.Binary

import Types


class Conductive s where
    current :: s -> Current


excitatory, inhibitory :: (Conductive s) => Synapse s -> Bool
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


instance (Binary s) => Binary (Synapse s) where
    put (Synapse s t d pl) = put s >> put t >> put d >> put pl
    get = liftM4 Synapse get get get get


instance Binary Static where
    put (Static w) = put w
    get = liftM Static get


instance (NFData s) => NFData (Synapse s) where
    rnf (Synapse s t d pl) = rnf s `seq` rnf t `seq` rnf d `seq` rnf pl


instance NFData Static where
    rnf (Static w) = rnf w


-- TODO: consitent name and argument order
retarget :: Synapse s -> Idx -> Synapse s
retarget (Synapse src tgt d pl) tgt' = Synapse src tgt' d pl


changeSource :: Idx -> Synapse s -> Synapse s
changeSource src' (Synapse src tgt d pl) = Synapse src' tgt d pl


mapIdx :: (Idx -> Idx) -> Synapse s -> Synapse s
mapIdx f (Synapse s t d pl) = Synapse (f s) (f t) d pl
