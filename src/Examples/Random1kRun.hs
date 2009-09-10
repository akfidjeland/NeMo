module Examples.Random1kRun (runRandom1kExample) where

import Examples.Random1k
import Construction.Izhikevich
import NSim
import Simulation.FiringStimulus


runRandom1kExample = do
    -- TODO: would like to express this in terms of neuron properties
    let ne = 800
    let ni = 200
    execute (random1k ne ni) NoFiring
