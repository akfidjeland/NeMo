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
import System.Environment (getArgs)
import System.Exit (exitWith, ExitCode(..))
import System.Time (getClockTime, ClockTime(..))
import System.IO (hPutStrLn, stderr)
import Text.Printf

import Construction.Connectivity
import Construction.Construction
import Construction.Network(printConnections, printNeurons, size)
import Construction.Parameterisation
import Construction.Randomised.Synapse
import Construction.Synapse
import Options
import Simulation.CUDA.Options
import Simulation.Run
import Simulation.STDP.Options (stdpOptions)
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
runSimulation seed simOpts net fstimF stdpOpts cudaOpts = do
    startConstruct <- getCPUTime
    net' <- buildNetwork seed net
    hPutStrLn stderr "Building simulation..."
    -- TODO: should use system time here instead
    putStrLn $ show $ size net'
    endConstruct <- getCPUTime
    hPutStrLn stderr $ "Building done (" ++ show(elapsed startConstruct endConstruct) ++ "s)"
    start <- getCPUTime
    -- TODO: use only a single probe function parameter
    runSim simOpts net' fstimF tsv cudaOpts stdpOpts
    end <- getCPUTime
    hPutStrLn stderr $ "Simulation done (" ++ show(elapsed start end) ++ "s)"
    where
        -- Return number of elapsed seconds since start
        elapsed start end = (end - start) `div` 1000000000000

        -- print firing information in tab-separated format
        tsv :: Int -> FiringOutput -> IO ()
        tsv t (FiringOutput xs) = mapM_ (\x -> printf "%u\t%u\n" t x) xs



{- Process externally defined network according to command-line options
 - (default to run forever). -}
execute net fstim = do
    (args, commonOpts) <- startOptProcessing =<< getArgs
    cudaOpts    <- processOptGroup cudaOptions args
    networkOpts <- processOptGroup (networkOptions FromCode) args
    stdpOpts    <- processOptGroup stdpOptions args
    simOpts     <- processOptGroup (simOptions AllBackends) args
    endOptProcessing args
    initRng $ optSeed commonOpts -- RNG for stimlulus
    processOutputOptions commonOpts networkOpts net
    -- TODO: use a single RNG? Currently one for build and one for stimulus
    execute_ commonOpts networkOpts simOpts stdpOpts cudaOpts net fstim



{- Process network provided from file according to command-line options -}
executeFile = do
    error "reading from file not currently supported"
{-
    (args, commonOpts) <- startOptProcessing
    cudaOpts    <- processOptGroup cudaOptions args
    networkOpts <- processOptGroup (networkOptions FromFile) args
    stdpOpts    <- processOptGroup stdpOptions args
    simOpts     <- processOptGroup (simOptions AllBackends)  args
    endOptProcessing args
    initRng $ optSeed commonOpts -- RNG for stimlulus
    let filename = fromMaybe (error "no file specified") $ optLoadNet networkOpts
    hPutStrLn stderr $ "Loading file from " ++ filename
    (net', fstim) <- decodeSimFile filename
    let net = return net' -- wrap in Gen
    processOutputOptions commonOpts networkOpts net
    execute_ commonOpts networkOpts simOpts stdpOpts cudaOpts net fstim
-}


{- | If requested, print network and terminate. Otherwise do nothing -}
processOutputOptions commonOpts networkOpts net
    | optDumpNeurons networkOpts = net' >>= printNeurons >> exitWith ExitSuccess
    | optDumpMatrix networkOpts  = net' >>= printConnections >> exitWith ExitSuccess
    | otherwise                 = return Nothing
    where
        net' = buildNetwork (optSeed commonOpts) net


execute_ commonOpts networkOpts simOpts stdpOpts cudaOpts net fstimF
    | optStoreNet networkOpts /= Nothing = error "serialisation to file not supported"
    | otherwise = runSimulation (optSeed commonOpts) simOpts net fstimF
                                stdpOpts cudaOpts
