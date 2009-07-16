{-# LANGUAGE CPP #-}

module Main where

import System.Console.GetOpt
import System.Environment (getArgs)
import System.Exit
import System.FilePath
import System.IO
import Test.HUnit

import Test.Construction.Axon (runQC)
import Test.Examples.Smallworld as Smallworld (tests)
import Test.Examples.Random1k as Random1k (tests)
import Test.Examples.Ring as Ring (tests)
import Test.Network.Client as TestClient (tests, test_clientSim)
import qualified Test.Network.ClientFFI (tests, create_tests)
import Test.Regression (testAll, createAll)
import Test.Simulation.Run as TestRun (tests)
#if defined(MATLAB_ENABLED)
import Test.ClientAPI.Matlab as Matlab (tests, create_tests)
#endif
#if defined(CUDA_ENABLED)
import Test.Simulation.CUDA.Mapping as Mapping (tests)
import Test.Simulation.CUDA.Memory as Memory (tests)
#endif

runQCTests = runQC

regressionTests = [Smallworld.tests, Random1k.tests]

-- | Run all HUnit tests
runHUnitTests dir = runTestTT $ TestList $ [
        TestRun.tests,
        Ring.tests,
        rtests,
#if defined(CUDA_ENABLED)
        Mapping.tests,
        Memory.tests,
#endif
        TestClient.tests,
#if defined(MATLAB_ENABLED)
        Matlab.tests,
#endif
        Test.Network.ClientFFI.tests dir
      ]
    where
        rtests = TestList $ map (testAll dir) regressionTests

createHUnitTests dir = do
    mapM_ (createAll dir) regressionTests
#if defined(MATLAB_ENABLED)
    Matlab.create_tests
#endif
    Test.Network.ClientFFI.create_tests dir


data Options = Options {
        optCreateRegressions :: Bool,
        optRegressionDir     :: String
    }

defaultOptions :: Options
defaultOptions = Options {
        optCreateRegressions = False,
        optRegressionDir = "testsuite" </> "regression-data"
    }


options :: [OptDescr (Options -> IO Options)]
options = [

    Option ['h'] ["help"]    (NoArg showHelp)
        "show command-line options",

    Option [] ["data-dir"]
        (ReqArg (\a o -> return o { optRegressionDir = a }) "DIR")
        ("base directory containing regression data. (default: " ++
            (optRegressionDir defaultOptions) ++ ")"),

    Option [] ["create-regression"]
        (NoArg (\o -> return o { optCreateRegressions = True }))
        "re-create regression data"
    ]

checkErrors []   = return ()
checkErrors msgs = do
    mapM (hPutStrLn stderr) msgs
    hPutStrLn stderr $ "Run runtests --help for summary of options"
    exitWith $ ExitFailure 1


showHelp _ = do
    putStr $ usageInfo "runtests [OPTIONS]\n\nOptions:" options
    exitWith ExitSuccess



main = do
    args <- getArgs
    let (actions, nonOpts, msgs) = getOpt RequireOrder options args
    checkErrors msgs
    opts <- foldl (>>=) (return defaultOptions) actions
    -- TestClient.test_clientSim
    if optCreateRegressions opts
        then createHUnitTests (optRegressionDir opts)
        else do
            runQCTests
            runHUnitTests (optRegressionDir opts)
            return ()
