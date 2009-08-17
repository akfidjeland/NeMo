module Test.Simulation.CUDA.Memory (tests) where

import Control.Monad.Writer (runWriter)
import Test.HUnit
import Foreign.C.Types (CFloat)
import Construction.Construction (build)
import Construction.Izhikevich (IzhNeuron)
import Construction.Network (Network, networkNeurons)
import Construction.Neurons (synapses, withSynapses, synapseCount, size)
import Construction.Synapse (Static(..))
import Examples.Smallworld (smallworld, smallworldOrig)
import Examples.Ring (ring)
import Options (defaults)
import Simulation.CUDA.Mapping (mapNetwork)
import Simulation.CUDA.Memory (initMemory, getWeights, SimData(..))
import Simulation.CUDA.KernelFFI (copyToDevice)
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
        net = build 123456 $ smallworldOrig :: Network (IzhNeuron FT) Static
        ns = withSynapses viaCFloat $ networkNeurons net
        ((cuNet, att), _) = runWriter $ mapNetwork net stdp psize
        nsteps = 1000 -- irrelevant for this test
    sim  <- initMemory cuNet att nsteps 4 (defaults stdpOptions)
    {- When initialising memory, the device may not be involved yet -}
    copyToDevice (rt sim)
    ns' <- getWeights sim

    assertEqual "Same number of neurons in weight matrix before and after writing to device"
       (size ns) (size ns')

    assertEqual
        "Same number of synapses in weight matrix before and after writing to device"
        (synapseCount ns) (synapseCount ns')

    {- The neuron will be different, since getWeights just puts dummy neurons
     - in the network. The synapses should be exactly the same, though -}
    assertEqual "Weight matrix before and after writing to device"
        (synapses ns) (synapses ns')

    where

        {- The weights we get back from the device may be slightly different
         - due to rounding to CFloat. Perform same operation on input neurons
         - as well -}
        viaCFloat (Static x) = Static $ realToFrac x'
            where x' = realToFrac x :: CFloat
