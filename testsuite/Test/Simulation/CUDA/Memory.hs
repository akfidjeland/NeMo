module Test.Simulation.CUDA.Memory (tests) where

import Control.Monad (zipWithM_)
import Control.Monad.Writer (runWriter)
import qualified Data.Map as Map
import Data.List (sort)
import Foreign.C.Types (CFloat)
import Test.HUnit

import Construction.Construction (build)
import Construction.Izhikevich (IzhNeuron)
import Construction.Network (Network, networkNeurons)
import Construction.Neurons (synapses, withWeights, synapseCount, size, weightMatrix)
import Construction.Synapse (Static(..))
import Examples.Smallworld (smallworld, smallworldOrig)
import Examples.Ring (ring)
import Options (defaults)
import Simulation.CUDA.Memory (initMemory, getWeights)
import Simulation.CUDA.State (State(..))
import Simulation.CUDA.KernelFFI (startSimulation)
import Simulation.STDP.Options (stdpOptions)
import Types (FT)


tests = TestLabel "CUDA memory" $ TestList [
            testWeightQuery
        ]

{- When reading weights back from the device, we should get back a network of
 - the same structure, although the weights may change -}
testWeightQuery = TestCase $ do
    let stdp = False
        psize = Just 128   -- to ensure different partitions are used
        net = build 123456 $ smallworldOrig :: Network IzhNeuron Static
        ns = weightMatrix $ withWeights viaCFloat $ networkNeurons net
        nsteps = 1000 -- irrelevant for this test
    sim  <- initMemory net psize nsteps 4 (defaults stdpOptions)
    {- When initialising memory, the device may not be involved yet -}
    startSimulation (rt sim)
    ns' <- getWeights sim

    assertEqual "Same number of neurons in weight matrix before and after writing to device"
       (Map.size ns) (Map.size ns')

    assertEqual
        "Same number of synapses in weight matrix before and after writing to device"
        (length $ concat $ Map.elems ns)
        (length $ concat $ Map.elems ns')

    {- The neuron will be different, since getWeights just puts dummy neurons
     - in the network. The synapses should be exactly the same, though -}
    -- TODO: compare this for every presynaptic
    let sorted = sort $ Map.assocs ns
    let sorted' = sort $ Map.assocs ns'
    zipWithM_ comparePresynaptic sorted sorted'

    where

        {- The weights we get back from the device may be slightly different
         - due to rounding to CFloat. Perform same operation on input neurons
         - as well -}
        viaCFloat :: Double -> Double
        viaCFloat x = realToFrac x'
            where x' = realToFrac x :: CFloat

        comparePresynaptic (k, ss) (k', ss') = do
            let sorted = sort ss
                sorted' = sort ss'
                msg = "Synapses for " ++ show k
            assertEqual "Presynaptic neuron " k k'
            zipWithM_ (assertEqual msg) sorted sorted'

