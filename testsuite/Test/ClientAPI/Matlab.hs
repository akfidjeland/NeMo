{-# LANGUAGE CPP #-}

module Test.ClientAPI.Matlab (tests, create_tests) where

import Control.Monad
import Control.Concurrent (forkIO, yield, throwTo)
import Control.Exception
import Data.List (isPrefixOf, intercalate)
import Network.BSD (getHostName)
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Process
import System.IO
import Test.HUnit

import Construction
import Network.Protocol (defaultPort)
import Network.Server (serveSimulation)
import Options
import Simulation.Run
import Simulation.Common
import Types


{- We use a regression test to verify that both the Matlab/Mex interface and
 - the socket communication works. -}
tests :: Test
tests = TestLabel "testing matlab API" $ TestCase $ do
    withSimulation $ do

    {- It seems there is no way to make matlab return a status code to the
     - shell, so to check whether the m-file was successful or not, we have to
     - rely on it returning a status string which we then have to parse.
     -
     - A return string starting with 'success' is deemed to be a success.
     - Anything else, a failure -}
    file <- dataFile
    results <- matlabRun (ml_runSmallworld ++ ml_checkFiring file)
    let status = last results
    assertBool status ("success" `isPrefixOf` status)


create_tests = withSimulation $ do
    outfile <- dataFile
    putStrLn $ "Writing regression data to " ++ outfile
    matlabRun $ ml_runSmallworld ++ ml_saveFiring outfile
    return ()


{- Run a sequence of commands in a matlab session, returning the matlab
 - response for each of them -}
matlabRun :: [String] -> IO [String]
matlabRun commands = do
    (cmd, out, _, _) <- runInteractiveProcess "matlab" ["-nosplash", "-nodesktop"] Nothing Nothing
    matlabOutput out -- skip the initial crud
    results <- mapM (exec cmd out) commands
    exec_ cmd "quit"
    return $ results
    where
        exec_ cmd s = hPutStrLn cmd s >> hFlush cmd      -- ignore output
        exec cmd out s = exec_ cmd s >> matlabOutput out -- return output


{- Read matlab outupt until the next prompt -}
matlabOutput :: Handle -> IO String
matlabOutput h = go h '\0' []
    where
        go h p a = do
            n <- hGetChar h
            if ">>" == [p,n]
                then do
                    hGetChar h  -- read the space after the prompt
                    return $! reverse $ drop 1 a
                else go h n (n:a)


{- Set up and run simulation, returning firing vector in 'f' -}
ml_runSmallworld = [
        "addpath('client-dist/matlab/latest');",
        "addpath('testsuite/Test/ClientAPI');",   -- for smallworld script
        "nsSetHost('localhost');",
        "nsSetPort(" ++ testPort ++ ");",
        "nsDisableSTDP;",
        "[a, b, c, d, u, v, post, delays, weights] = smallworld(34, 12345);",
        "nsStart(a, b, c, d, u, v, post, delays, weights, 4, 1);",
        "[f, t] = nsRun(1000, [1, 1]);",
        "nsTerminate;"
    ]


{- Write firing results to file -}
ml_saveFiring filename = [
        "csvwrite('" ++ filename ++ "', f);"
    ]


{- Load firing results from file and compare with data in 'f' -}
ml_checkFiring filename = [
        "f_check = csvread('" ++ filename ++ "');",
        intercalate "\n" [
            "if isequal(f, f_check) ~= 1",
            "   disp 'failure: expected firing did not match actual firing'",
            "else",
            "   disp 'success: firing just as expected'",
            "end"
          ]
    ]



dataFile = do
    hostname <- getHostName
    return $! "testsuite" </> "regression-data" </> hostname </> "smallworld-matlab-1000.dat"



{- Make sure to run on non-standard port to avoid interference with running server -}
testPort = show $ defaultPort + 2

withSimulation f = bracket (forkIO runServer) kill (\_ -> f)
    where
        {- Killing the server is a bit brutal, but UserInterrupt is "not in
         - scope" for some reason -}
        kill s = throwTo s $ AsyncException ThreadKilled
        -- kill s = throwTo s $ AsyncException UserInterrupt


runServer :: IO ()
runServer = do
    let logdir = "testsuite" </> "log"
    createDirectoryIfMissing True logdir
    let logfile = logdir </> "test_matlabAPI.log"
    bracket (openFile logfile WriteMode) hClose $ \hdl -> do
    serveSimulation hdl testPort False
    -- serveSimulation stdout testPort False
        (\net tr -> initSim CPU
                    (net :: Network (IzhNeuron FT) Static)
                    All
                    (Firing :: ProbeFn IzhState)
                    tr False defaultOptions)
