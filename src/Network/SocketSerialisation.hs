{- Serialised data are sent over a socket interface. This is done by chunks, so
 - we can deal with arbitrarily large data without running into buffering
 - problems. -}
module Network.SocketSerialisation (
    sendSerialised,
    recvSerialised
) where

import Data.Binary
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as BL
import Network.Socket hiding (send, sendTo, recv, recvFrom)
import Network.Socket.ByteString


-- | Receive a packet in number of chunks and deserialise it
recvSerialised :: (Binary a) => Socket -> IO a
recvSerialised sock = return . decode =<< getBS sock


-- | Serialise data and send over socket in several chunks
sendSerialised :: (Binary a) => Socket -> a -> IO ()
sendSerialised sock packet = do
    let chunks = BL.toChunks $ encode packet
    mapM_ (sendChunk sock) $ chunks
    sendNullChunk sock
    where
        sendNullChunk sock = sendChunkHeader sock 0 >> return ()


-- | Convert lazy bytestring to strict
strictifyBS :: BL.ByteString -> B.ByteString
strictifyBS = B.concat . BL.toChunks


{- | Haskell sockets default to non-blocking and there's no straightforward way
 - to change this. The reason for this is that the RTS may have several active
 - threads, and that blocking would block all threads. -}
sendBlocking :: Socket -> B.ByteString -> IO ()
sendBlocking sock payload = do
    sent <- send sock payload
    {- low-level errors result in an IOError exception, rather than a return
     - status -}
    if sent == B.length payload
        then return ()
        else sendBlocking sock $ B.drop sent payload


-- | Send chunk header (length of chunk)
sendChunkHeader sock len = do
    let bytes = strictifyBS $ encode (len :: Int)
    sendBlocking sock $ strictifyBS $ encode (len :: Int)


-- | Send a single chunk of data
sendChunk :: Socket -> B.ByteString -> IO ()
sendChunk sock chunk = do
    let chunkLen = B.length chunk -- O(1)
    sendChunkHeader sock chunkLen
    sendBlocking sock chunk


-- | Receive a single chunk of known length as a lazy bytestring
recvChunk :: Socket -> Int -> IO BL.ByteString
recvChunk sock len = return . BL.fromChunks =<< recvChunk_ sock len
    where
        recvChunk_ :: Socket -> Int -> IO [B.ByteString]
        recvChunk_ sock len = do
            string <- recv sock len
            case len - B.length string of
                0 -> return $! [string]
                remaining -> do
                     rest <- recvChunk_ sock remaining
                     return $! string : rest


-- | Receive a bytestring in several chunks
getBS :: Socket -> IO BL.ByteString
getBS sock = do
    header <- recvChunk sock 8
    case (decode header) :: Int of
        0   -> do
            return $! BL.empty
        len -> do
            h <- recvChunk sock len
            t <- getBS sock
            return $! h `BL.append` t
