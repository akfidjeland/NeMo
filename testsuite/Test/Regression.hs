module Test.Regression (
    RegressionTest(..),
    testAll,
    createAll
) where

import Data.List (sort)
import Network.BSD (getHostName)
import System.Directory (createDirectoryIfMissing)
import System.FilePath (takeDirectory, (</>))
import System.IO
import System.Random
import Test.HUnit
import Test.QuickCheck

import Construction.Construction (build)
import Construction.Network
import Construction.Izhikevich
import Construction.Synapse
import Options (defaultOptions)
import Simulation.Common hiding (cycles)
import Simulation.FiringStimulus
import Simulation.Run (runSim)
import Simulation.STDP
import Types


data RegressionTest = RegressionTest {
        name     :: String,
        dataFile :: FilePath,
        netGen   :: Gen (Network (IzhNeuron FT) Static),
        fstim    :: FiringStimulus,
        probe    :: Probe,
        probeFn  :: ProbeFn IzhState,
        backend  :: Backend,
        cycles   :: Time,
        rngSeed  :: Int,
        stdp     :: Maybe Int  -- period with which STDP should be applied
    }


type SimulationHandler = (ProbeData -> IO ()) -> IO ()

createAll :: FilePath -> [RegressionTest] -> IO ()
createAll basedir rs = mapM_ (createRegression basedir) rs

testAll :: FilePath -> [RegressionTest] -> Test
testAll basedir rs = TestList $ map f rs
    where
        f r = TestLabel (name r) $ TestCase $ testRegression basedir r



createRegression :: FilePath -> RegressionTest -> Assertion
createRegression basedir rtest = do
    filename <- testFile basedir $ dataFile rtest
    createDirectoryIfMissing True $ takeDirectory filename
    outfile <- openFile filename WriteMode
    putStrLn $ "Create regression data: " ++ filename
    runRegression rtest $ hPutStrLn outfile . show
    -- TODO: use handle or similar to ensure we always close
    hClose outfile


testRegression :: FilePath -> RegressionTest -> Assertion
testRegression basedir rtest = do
    filename <- testFile basedir $ dataFile rtest
    infile <- openFile filename ReadMode
    runRegression rtest $ checkResults infile
    hClose infile


runRegression :: RegressionTest -> (ProbeData -> IO ()) -> Assertion
runRegression rtest@(RegressionTest nm _ nf fs px pf be cs rs stdp) f = do
    -- TODO: share code with NSim/Simulation.Run
    -- TODO: don't use the same seed twice. Also sync with use in NSim/Construction
    let net = build (fromIntegral rs) nf
    setStdGen $ mkStdGen $ rs -- for stimulus
    runSim be (Until cs) net px pf 4 fs f defaultOptions stdpConf
    where
        stdpConf = stdp >>= return . (STDPConf 20 20 1.0 0.8 1000.0) . Just


-- Run regression and dump data to file
create_regression :: FilePath -> String -> SimulationHandler -> IO ()
create_regression basedir testname run = do
    filename <- testFile basedir testname
    outfile <- openFile filename WriteMode
    run (hPutStrLn outfile . show)
    -- TODO: use handle or similar to ensure we always close
    hClose outfile


test_regression :: FilePath -> String -> SimulationHandler -> IO ()
test_regression basedir testname run = do
    filename <- testFile basedir testname
    -- TODO: missing data should result in a failure
    infile <- openFile filename ReadMode
    run (checkResults infile)
    hClose infile


testFile :: FilePath -> String -> IO FilePath
testFile basedir testname = do
    hostname <- getHostName
    return $ basedir </> hostname </> testname


checkResults :: Handle -> ProbeData -> IO ()
checkResults hdl (FiringData probeData) = do
    line <- hGetLine hdl
    -- let testData = FiringData $ sort $ read line
    let testData = sort $ read line
    assertEqual "" testData $ sort probeData
checkResults hdl _ = error "Non-firing data"
