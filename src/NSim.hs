module NSim (execute,
    Gen,
    mkRSynapse, mkSynapse,
    module Types,
    module Construction.Connectivity,
    module Construction.Parameterisation,
    module Construction.Construction,
    excitatory)
    where


import Control.Parallel.Strategies (($|), rnf)
import Data.Maybe
import Test.QuickCheck (Gen)
import System.Random (mkStdGen, setStdGen)
import CPUTime (getCPUTime)
import System.Time (getClockTime, ClockTime(..))
import System.IO (hPutStrLn, stderr)

import Construction.Connectivity
import Construction.Construction
import Construction.Network(printConnections, printNeurons, size)
import Construction.Parameterisation
import Construction.Randomised.Synapse
import Construction.Synapse
import Options
import Simulation.Common
import Simulation.FileSerialisation (encodeSimFile)
import Simulation.Run
import Types




buildNetwork seed net = do
    (TOD sec psec) <- getClockTime
    let seed' = fromMaybe (sec+psec) seed
    -- return $ build seed' net
    let net' = build seed' net
    -- note: this does not seem to work as we expect
    return $ (id $| rnf) net'



-- initialise the global RNG
initRng :: Maybe Integer -> IO ()
initRng Nothing = return ()
initRng (Just seed) = setStdGen $ mkStdGen $ fromInteger seed



-- TODO: migrate to Simulation.Run
runSimulation seed backend duration net tempSubres fstimF probeIdx probeFn opts = do
    startConstruct <- getCPUTime
    net' <- buildNetwork seed net
    hPutStrLn stderr "Building simulation..."
    -- TODO: should use system time here instead
    putStrLn $ show $ size net'
    endConstruct <- getCPUTime
    hPutStrLn stderr $ "Building done (" ++ show(elapsed startConstruct endConstruct) ++ "s)"
    start <- getCPUTime
    -- TODO: use only a single probe function parameter
    runSim backend duration net' probeIdx probeFn tempSubres fstimF (putStrLn . show) opts (stdpOptions opts)
    end <- getCPUTime
    hPutStrLn stderr $ "Simulation done (" ++ show(elapsed start end) ++ "s)"
    where
        -- Return number of elapsed seconds since start
        elapsed start end = (end - start) `div` 1000000000000



-- TODO: generalise to other neuron types. Requires making backends general.
execute progname net fstim probeidx probefn = do
    opts <- getOptions progname defaultOptions $
            commonOptions ++
            verbosityOptions ++
            storeOptions ++
            backendOptions ++
            simOptions ++
            printOptions
    -- TODO: use a single RNG? Currently one for build and one for stimulus
    initRng $ optSeed opts -- RNG for stimlulus
    execute_ opts net fstim probeidx probefn


execute_ opts net fstimF probeidx probefn
    | optStoreNet opts /= Nothing = do
        net' <- buildNetwork (optSeed opts) net
        encodeSimFile (fromJust $ optStoreNet opts) net' fstimF
    | optDumpNeurons opts = buildNetwork (optSeed opts) net >>= printNeurons
    | optDumpMatrix opts  = buildNetwork (optSeed opts) net >>= printConnections
    | otherwise           = runSimulation
                                (optSeed opts)
                                (optBackend opts)
                                (optDuration opts)
                                net (optTempSubres opts)
                                fstimF
                                probeidx probefn
                                opts
