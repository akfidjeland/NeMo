module Simulation.FileSerialisation (
    encodeNetworkFile,
    decodeNetworkFile,
    encodeSimFile,
    decodeSimFile
) where

import Control.Exception (bracket)
import Control.Monad (liftM2)
import Data.Binary (Binary, put, get, encode, decode)
import qualified Data.ByteString.Lazy as BL (hGetContents, hPutStr)
import System.IO

import Construction.Network (Network)
import Simulation.FiringStimulus

{- | Magic number that identifies the file encoding. 'decode' tends to happily
 - consume garbage data untill it runs out of stack-space, so its's desirable
 - to detect invalid data early. This magic number should be modified if the
 - serialisation routines are changed. -}
type MagicNumber = String

simMagic :: MagicNumber
simMagic = "NSIM001SIM"

netMagic :: MagicNumber
netMagic = "NSIM001NET"


data SimulationState n s = SimulationState {
        net   :: Network n s,
        fstim :: FiringStimulus
    }

instance (Binary n, Binary s) => Binary (SimulationState n s) where
    put (SimulationState net f) = put net >> put f
    get = liftM2 SimulationState get get


encodeFile :: (Binary a) => MagicNumber -> FilePath -> a -> IO ()
encodeFile magic path xs = do
    bracket (openFile path WriteMode) hClose $ \hdl -> do
    mapM_ (hPutChar hdl) magic
    BL.hPutStr hdl $ encode $ xs


decodeFile :: (Binary a) => MagicNumber -> FilePath -> IO a
decodeFile expectedMagic path = do
    hdl <- openFile path ReadMode
    checkMagic hdl expectedMagic
    return . decode =<< BL.hGetContents hdl
    where
        checkMagic hdl expectedMagic = do
            foundMagic <- sequence $ replicate (length expectedMagic) (hGetChar hdl)
            if foundMagic == expectedMagic
                then return ()
                else fail $ "Wrong magic number found when decoding file, expected " 
                    ++ expectedMagic ++ ", found " ++ foundMagic


encodeSimFile :: (Binary n, Binary s)
    => FilePath -> Network n s -> FiringStimulus -> IO ()
encodeSimFile path net fstim =
    encodeFile simMagic path $ SimulationState net fstim


decodeSimFile :: (Binary n, Binary s)
    => FilePath -> IO (Network n s, FiringStimulus)
decodeSimFile path = do
    sim <- decodeFile simMagic path
    return (net sim, fstim sim)


encodeNetworkFile :: (Binary n, Binary s) => FilePath -> Network n s -> IO ()
encodeNetworkFile = encodeFile netMagic


decodeNetworkFile :: (Binary n, Binary s) => FilePath -> IO (Network n s)
decodeNetworkFile = decodeFile netMagic
