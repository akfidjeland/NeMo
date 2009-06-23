module Test.Regression (
    RegressionTest(..),
    testAll,
    createAll
) where

import Data.List (sort)
import Data.Maybe (isJust)
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
import Options (defaults)
import Simulation.Common hiding (cycles)
import Simulation.CUDA.Options (cudaOptions)
import Simulation.FiringStimulus
import Simulation.Options (simOptions, optBackend, optDuration, BackendOptions(..))
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
    runSim simOpts net px pf fs f (defaults cudaOptions) stdpConf
    where
        stdpConf = STDPConf (isJust stdp) 20 20 1.0 0.8 1000.0 stdp
        simOpts  = (defaults $ simOptions LocalBackends) {
                optBackend = be,
                optDuration = Until cs
            }


testFile :: FilePath -> String -> IO FilePath
testFile basedir testname = do
    hostname <- getHostName
    return $ basedir </> hostname </> testname


checkResults :: Handle -> ProbeData -> IO ()
checkResults hdl (FiringData probeData) = do
    line <- hGetLine hdl
    let testData = sort $ read line
    assertEqual "" testData $ sort probeData
checkResults hdl _ = error "Non-firing data"
