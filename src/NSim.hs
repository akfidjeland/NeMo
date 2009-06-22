module NSim (
    execute,
    executeFile,
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
import System.Exit (exitWith, ExitCode(..))
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
import Simulation.CUDA.Options
import Simulation.FileSerialisation (encodeSimFile, decodeSimFile)
import Simulation.Run
import Simulation.STDP (stdpOptions)
import Simulation.Options (simOptions, optBackend, BackendOptions(..))
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
runSimulation seed simOpts net fstimF probeIdx probeFn stdpOpts cudaOpts = do
    startConstruct <- getCPUTime
    net' <- buildNetwork seed net
    hPutStrLn stderr "Building simulation..."
    -- TODO: should use system time here instead
    putStrLn $ show $ size net'
    endConstruct <- getCPUTime
    hPutStrLn stderr $ "Building done (" ++ show(elapsed startConstruct endConstruct) ++ "s)"
    start <- getCPUTime
    -- TODO: use only a single probe function parameter
    runSim simOpts net' probeIdx probeFn fstimF (putStrLn . show) cudaOpts stdpOpts
    end <- getCPUTime
    hPutStrLn stderr $ "Simulation done (" ++ show(elapsed start end) ++ "s)"
    where
        -- Return number of elapsed seconds since start
        elapsed start end = (end - start) `div` 1000000000000



{- Process externally defined network according to command-line options
 - (default to run forever). -}
execute net fstim probeidx probefn = do
    (args, commonOpts) <- startOptProcessing
    cudaOpts    <- processOptGroup cudaOptions args
    networkOpts <- processOptGroup (networkOptions FromCode) args
    stdpOpts    <- processOptGroup stdpOptions args
    simOpts     <- processOptGroup (simOptions ClientBackends) args
    endOptProcessing args
    initRng $ optSeed commonOpts -- RNG for stimlulus
    processOutputOptions commonOpts networkOpts net
    -- TODO: use a single RNG? Currently one for build and one for stimulus
    execute_ commonOpts networkOpts simOpts stdpOpts cudaOpts net fstim probeidx probefn



{- Process network provided from file according to command-line options -}
executeFile = do
    (args, commonOpts) <- startOptProcessing
    cudaOpts    <- processOptGroup cudaOptions args
    networkOpts <- processOptGroup (networkOptions FromFile) args
    stdpOpts    <- processOptGroup stdpOptions args
    simOpts     <- processOptGroup (simOptions ClientBackends)  args
    endOptProcessing args
    initRng $ optSeed commonOpts -- RNG for stimlulus
    let filename = fromMaybe (error "no file specified") $ optLoadNet networkOpts
    hPutStrLn stderr $ "Loading file from " ++ filename
    (net', fstim) <- decodeSimFile filename
    let net = return net' -- wrap in Gen
    processOutputOptions commonOpts networkOpts net
    execute_ commonOpts networkOpts simOpts stdpOpts cudaOpts net fstim All Firing



{- | If requested, print network and terminate. Otherwise do nothing -}
processOutputOptions commonOpts networkOpts net
    | optDumpNeurons networkOpts = net' >>= printNeurons >> exitWith ExitSuccess
    | optDumpMatrix networkOpts  = net' >>= printConnections >> exitWith ExitSuccess
    | otherwise                 = return Nothing
    where
        net' = buildNetwork (optSeed commonOpts) net


execute_ commonOpts networkOpts simOpts stdpOpts cudaOpts
    net fstimF probeidx probefn
    | optStoreNet networkOpts /= Nothing = do
        net' <- buildNetwork (optSeed commonOpts) net
        encodeSimFile (fromJust $ optStoreNet networkOpts) net' fstimF
    | otherwise = runSimulation (optSeed commonOpts) simOpts net fstimF
                                probeidx probefn stdpOpts cudaOpts
