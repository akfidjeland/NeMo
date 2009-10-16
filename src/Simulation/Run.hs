{- Do a complete simulation run -}

module Simulation.Run (runSim) where

import Control.Monad (forM, mapM_, zipWithM, zipWithM_, when)
import Data.Maybe (isJust)

import Simulation (Simulation_Iface(..))
import Simulation.Backend (initSim)
import Simulation.FiringStimulus (firingStimulus)
import Simulation.Options
import Simulation.STDP (stdpFrequency)
import qualified Util.List as L (chunksOf)
import Types


{- | Run full simulation using the appropriate backend -}
runSim simOpts net fstimF outfn opts stdpConf = do
    fstim <- firingStimulus fstimF
    sim <- initSim net simOpts opts stdpConf
    let fs = L.chunksOf chunkSize $ sample duration fstim
    let ts = L.chunksOf chunkSize [0..]
    zipWithM (go sim) fs ts
    stop sim
    where
        duration = optDuration simOpts
        freq = stdpFrequency stdpConf
        chunkSize = maybe 1000 id freq

        go sim fs ts = do
            probed <- run sim fs
            when (isJust freq) $ applyStdp sim 1.0
            zipWithM_ outfn ts probed

        {- | Apply function for a given duration -}
        sample :: Duration -> [a] -> [a]
        sample Forever xs = xs
        sample (Until end) xs = take end xs

