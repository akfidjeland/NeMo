module Test.Simulation.CUDA.Memory (tests) where

import Control.Monad (zipWithM_)
import Control.Monad.Writer (runWriter)
import Data.Function (on)
import Data.List (sort)
import qualified Data.Map as Map
import Foreign.C.Types (CFloat)
import Test.HUnit

import Construction.Construction (build)
import Construction.Izhikevich (IzhNeuron)
import Construction.Network (Network, networkNeurons)
import Construction.Neurons (synapses, withWeights, synapseCount, size, weightMatrix)
import Construction.Synapse (Static(..), Synaptic(..))
import Examples.Smallworld (smallworld, smallworldOrig)
import Examples.Ring (ring)
import Options (defaults)
import Simulation.CUDA.Memory (initSim, getWeights, rtdata)
import Simulation.CUDA.KernelFFI (initSimulation)
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
    sim  <- initSim net psize (defaults stdpOptions)
    {- When initialising memory, the device may not be involved yet -}
    initSimulation $ rtdata sim
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

        {- Additionally, we cast to fixed-point format and back again, which
         - may give different results. We just test that it's not too different -}
        weightsClose :: Double -> Double -> Bool
        weightsClose w1 w2 = w1 - w2 < 0.00001

        checkSynapses :: (Synaptic s) => String -> s -> s -> Assertion
        checkSynapses msg s1 s2 = do
            let check fieldname field = assertEqual (msg ++ fieldname) (field s1) (field s2)
            check " target" target
            check " delay" delay
            check " plastic" plastic
            assertBool (msg ++ " weight") $ (weightsClose `on` weight) s1 s2

        comparePresynaptic (k, ss) (k', ss') = do
            let sorted = sort ss
                sorted' = sort ss'
                msg = "Synapses for " ++ show k
            assertEqual "Presynaptic neuron " k k'
            zipWithM_ (checkSynapses msg) sorted sorted'
