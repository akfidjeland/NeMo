module Construction.Randomised.Synapse (mkSynapse, mkRSynapse) where

import Test.QuickCheck

import Construction.Synapse (Synapse(..), Static(..))
import Types

-- TODO: just switch the parameters for ctor
{- Randomisable synapse -}
mkRSynapse :: Gen FT -> Gen Time -> Idx -> Idx -> Gen (Synapse Static)
mkRSynapse w d pre post = do
    w' <- w
    d' <- d
    let s = w' `seq` d' `seq` Synapse pre post d' w' ()
    return s


{- Synapse generator with fixed parameters -}
mkSynapse :: FT -> Time -> Idx -> Idx -> Gen (Synapse Static)
mkSynapse w d pre post = return $! Synapse pre post d w ()
