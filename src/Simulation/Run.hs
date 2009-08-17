{- Common simulation interface for different backends -}
module Simulation.Run (runSim) where

import Control.Monad (forM, mapM_, when)
import Data.Maybe (isJust)

import Simulation (Simulation_Iface(..))
import Simulation.Backend (initSim)
import Simulation.FiringStimulus (firingStimulus)
import Simulation.Options
import Simulation.STDP (stdpFrequency)
import qualified Util.List as L (chunksOf)
import Types


{- | Run full simulation using the appropriate backend -}
-- TODO: fix argument ordering
runSim simOpts net fstimF outfn opts stdpConf = do
    fstim <- firingStimulus fstimF
    sim <- initSim net simOpts opts stdpConf
    go sim $ sample duration fstim
    terminate sim
    where
        duration = optDuration simOpts
        freq = stdpFrequency stdpConf
        chunkSize = maybe 1000 id freq

        go sim fstim =
            forM (L.chunksOf chunkSize fstim) $ \f -> do
            probed <- run sim f
            when (isJust freq) $ applyStdp sim 1.0
            mapM_ outfn probed

        {- | Apply function for a given duration -}
        sample :: Duration -> [a] -> [a]
        sample Forever xs = xs
        sample (Until end) xs = take end xs

