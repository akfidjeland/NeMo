{- | Generic synapse
 -
 - Due to memory issues synapses are generally stored in more compact formats
 - internally (see "Construction.Axon").
 -}

module Construction.Synapse (
        Synapse(..),
        Static,
        excitatory,
        inhibitory,
        retarget
    ) where

import Types (Source, Target, Delay, Weight, Idx)


{- | In addition to the four main fields of the synapse there is an auxillary
 - "payload" field which can store additional data, such as plasticity
 - parameters, etc -}
data Synapse s = Synapse {
        source :: {-# UNPACK #-} !Source,
        target :: {-# UNPACK #-} !Target,
        delay  :: {-# UNPACK #-} !Delay,
        weight :: {-# UNPACK #-} !Weight,
        sdata  :: {-# UNPACK #-} !s
    } deriving (Eq, Show, Ord)


type Static = ()


excitatory, inhibitory :: Synapse s -> Bool
excitatory s = weight s > 0
inhibitory s = weight s < 0


retarget :: Target -> Synapse s -> Synapse s
retarget tgt' (Synapse src tgt d w pl) = Synapse src tgt' d w pl
