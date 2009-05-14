module Test.Network.ClientFFI (tests, create_tests) where

import Foreign.Marshal.Array
import Foreign.C.Types
import System.FilePath
import Test.HUnit

import Construction
import Network.ClientFFI
import Simulation.FileSerialisation (encodeNetworkFile, decodeNetworkFile)
import Types

tests dir = TestList [
        TestLabel "Marshalling network in FFI interface"
            (TestCase $ test_marshalling dir)
    ]


create_tests dir = do
    create_marshalling dir


{- Check that network marshalling always does the same -}
test_marshalling :: FilePath -> Assertion
test_marshalling dir = do
    sw0 <- decodeNetworkFile (filename dir)
    sw1 <- testnet
    assertEqual [] sw0 sw1


create_marshalling :: FilePath -> IO ()
create_marshalling dir = encodeNetworkFile (filename dir) =<< testnet


filename :: FilePath -> FilePath
filename dir = dir </> "Network-ClientFFI-Marshalling" <.> "dat"


{- Create network using marshalling routine -}
testnet :: IO (Network (IzhNeuron FT) Static)
testnet = do
    let n = 1000
        m = 100
    a <- newArray $ take n list0
    b <- newArray $ take n list0
    c <- newArray $ take n list0
    d <- newArray $ take n list0
    u <- newArray $ take n list0
    v <- newArray $ take n list0
    sidx <- newArray $ take (n*m) $ concat $ replicate n ([0..m-1] :: [CInt])
    sdelay <- newArray $ take (n*m) $ repeat (20 :: CInt)
    sweight <- newArray $ take (n*m) $ concat $ replicate n ([-50.0..] :: [CDouble])
    createNetwork n m a b c d u v sidx sdelay sweight
    where
        list0 :: [CDouble]
        list0 = [0.0..]
