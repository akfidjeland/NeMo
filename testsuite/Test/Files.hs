module Test.Files (openTemp, closeTemp) where

import System.Directory
    (createDirectoryIfMissing, getTemporaryDirectory, removeFile)
import System.FilePath ((</>))
import System.IO (openTempFile, hClose)

openTemp name = do
    base <- getTemporaryDirectory
    let dir = base </> "nemo"
    createDirectoryIfMissing True dir
    openTempFile dir name

closeTemp (f, h) = hClose h >> removeFile f
