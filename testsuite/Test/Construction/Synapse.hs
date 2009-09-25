{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}

module Test.Construction.Synapse where

import Test.QuickCheck

import Construction.Synapse


{- The implicit size parameter is interpreted as the upper bound on neuron
 - indices. -}
instance Arbitrary (Synapse Static) where
    arbitrary = sized $ \mx -> do
        src <- choose (0, mx-1)
        term <- resize mx arbitrary
        return $! unstrip src term



{- The implicit size parameter is interpreted as the upper bound on neuron
 - indices. -}
instance Arbitrary (AxonTerminal Static) where
    arbitrary = sized $ \mx -> do
        tgt <- choose (0, mx-1)
        {- 32 is maximum delay imposed by CUDA backend -}
        d <- choose (1,32)
        {- The weight is unlikely to be relevant in quickcheck testing, as it
         - does not affect mapping etc, just execution (which is tested using
         - known networks). -}
        w <- choose (-1.0, 1.0)
        return $! AxonTerminal tgt d w ()
