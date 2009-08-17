{- | Compare two simulation runs (data written to temporary files). -}
module Test.Comparative (comparisonTest, compareSims) where

import Control.Exception (bracket)
import System.IO
import Test.HUnit

import Test.Files (openTemp, closeTemp)
import Types

type HSimulation = (ProbeData -> IO ()) -> IO ()

comparisonTest :: HSimulation -> HSimulation -> String -> Test
comparisonTest sim1 sim2 msg = TestLabel msg $ TestCase $ compareSims sim1 sim2


compareSims :: HSimulation -> HSimulation -> Assertion
compareSims sim1 sim2 = do
    bracket (openTemp "test_compare") closeTemp $ \(_, h1) -> do
    bracket (openTemp "test_compare") closeTemp $ \(_, h2) -> do
    sim1 $ hPutStrLn h1 . show
    sim2 $ hPutStrLn h2 . show
    hSeek h1 AbsoluteSeek 0
    hSeek h2 AbsoluteSeek 0
    c1 <- hGetContents h1
    c2 <- hGetContents h2
    let cmp = zipWith (==) c1 c2
    if all id cmp
        then return ()
        else assertFailure $
            "Simulation output mismatch. First error in byte " ++
            (show $ fst $ head $ dropWhile snd $ zip [0..] cmp) ++
            " out of " ++ (show $ length cmp)
