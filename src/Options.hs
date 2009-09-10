{-# LANGUAGE CPP #-}

{- All options used by *any* of the programs. Individual programs may use only
 - a subset of these, or use some different default. -}

module Options (
    -- * General option processing
    startOptProcessing,
    processOptGroup,
    endOptProcessing,
    -- * Option definition
    OptionGroup(..),
    withDefault,
    OptDescr(..),
    ArgDescr(..),
    -- * Common options
        optVerbose,
        optSeed,
    -- * Load/store options
    networkOptions,
        optDumpNeurons,
        optDumpMatrix,
        optStoreNet,
        optLoadNet,
    NetworkSource(..)
) where


import Control.Monad.Error
import Data.Either
import Data.Maybe
import Data.List (intercalate)
import Data.IORef
import System.Environment (getArgs, getProgName)
import System.Console.GetOpt
import System.Exit
import System.IO (hPutStrLn, stderr)

import Types


data OptionGroup g = OptionGroup {
        groupName :: String,
        defaults :: g,
        descr    :: [OptDescr (g -> Either String g)]
    }


optionGroupUsage :: OptionGroup g -> String
optionGroupUsage group = usageInfo (groupName group) (descr group)


{- | Annotate option description string with a default value -}
withDefault :: (Show a) => a -> String -> String
withDefault def descr = descr ++ " (default: " ++ show def ++ ")"


{- Initialise data for option processing. The processing takes place in one of
 - two modes. In help printing mode, usage for all options are printed, but no
 - parsing takes place. In parsing mode options are parsed and written to
 - option group data -}
startOptProcessing args = do
    ref <- newIORef args
    -- always process common options
    commonOpts <- processOptGroup commonOptions $ Just ref
    if optShowHelp commonOpts
        then do
            -- reprocess common options in help mode
            processOptGroup commonOptions Nothing
            return (Nothing, commonOpts)
        else return (Just ref, commonOpts)


{- Either parse options or print help message for the group -}
processOptGroup :: OptionGroup g -> Maybe (IORef [String]) -> IO g
processOptGroup group argref = do
    maybe
        (do putStrLn $ optionGroupUsage group
            return $ defaults group) -- for type checking, won't be used
        (parseOptGroup group)
        argref


{- Parse options from a group -}
parseOptGroup :: OptionGroup g -> IORef [String] -> IO g
parseOptGroup group argref = do
    args <- readIORef argref
    let (actions, nonOpts, args', msgs) = getOpt' Permute (descr group) args
    writeIORef argref args'
    if null msgs
        then do
            let opts = foldl (>>=) (return (defaults group)) actions
            either die return opts
        else do
            hPutStrLn stderr "an error occurred"
            mapM_ (hPutStrLn stderr) msgs
            exitWith $ ExitFailure 1
    where
        die msg = do
            hPutStrLn stderr msg
            exitWith $ ExitFailure 1


{- | Print error message and terminate if there where unknown errors on the
 - command-line -}
handleUnknownOpts :: IORef [String] -> IO ()
handleUnknownOpts args = do
    unknown <- readIORef args
    if null unknown
        then return ()
        else do
            progname <- getProgName
            hPutStrLn stderr $ "Unknown " ++ pluralise (length unknown) "option"
                ++ ":\n\t" ++ intercalate "\n\t" unknown
                ++ "\nRun " ++ progname ++ " --help for summary of options"
            exitWith $ ExitFailure 1
    where
        pluralise 1 str = str
        pluralise _ str = str ++ "s"


{- | If we're in help printing mode, terminate program. Otherwise report any
 - errors regarding unknown options -}
endOptProcessing :: Maybe (IORef [String]) -> IO ()
endOptProcessing =
    maybe
        (exitWith $ ExitSuccess) -- should have printed all help by now
        handleUnknownOpts


commonOptions = OptionGroup "Common options" commonDefaults commonDescr

data CommonOptions = CommonOptions {
        optVerbose  :: Bool,
        optShowHelp :: Bool,
        optSeed     :: Maybe Integer

    }

commonDefaults = CommonOptions {
        optVerbose  = False,
        optShowHelp = False,
        optSeed     = Nothing
    }


commonDescr = [
        Option ['h'] ["help"]
            (NoArg (\o -> return o { optShowHelp = True }))
            "show command-line options",

        Option ['v'] ["verbose"]
            (NoArg (\o -> return o { optVerbose = True }))
            "more than usually verbose output",

        Option ['s'] ["seed"]    (ReqArg readSeed "INT")
            "seed for random number generation (default: system time)"
    ]

readSeed arg opt = return opt { optSeed = Just $ read arg }



data NetworkOptions = NetworkOptions {
        optDumpNeurons :: Bool,
        optDumpMatrix  :: Bool,
        optLoadNet     :: Maybe String,
        optStoreNet    :: Maybe String
    }


networkDefaults = NetworkOptions {
        optDumpNeurons = False,
        optDumpMatrix  = False,
        optLoadNet     = Nothing,
        optStoreNet    = Nothing
    }


{- We can load network from file and also write to a file. It makes little
 - sense to load a network from file only to write it back to another file, so
 - we operate in one of two modes: -}
data NetworkSource
    = FromFile  -- ^ require network to be specified in file
    | FromCode  -- ^ get network internally, allow dump to file


networkOptions :: NetworkSource -> OptionGroup NetworkOptions
networkOptions io =
    OptionGroup "Output options" networkDefaults (networkDescr io)


networkDescr io = baseOpts ++ case io of
                                FromFile -> loadOpts
                                FromCode -> storeOpts
    where
        baseOpts = [
            Option ['C'] ["connectivity"]
                (NoArg (\o -> return o { optDumpMatrix=True }))
                "instead of simulating, just dump the connectivity matrix",

            Option ['N'] ["neurons"]
                (NoArg (\o -> return o { optDumpNeurons=True }))
                "instead of simulating, just dump the list of neurons"
          ]

        loadOpts = [
            Option [] ["load-network"]
                (ReqArg (\a o -> return o { optLoadNet = Just a }) "FILE")
                "load network from file"
          ]

        storeOpts = [
            Option [] ["store-network"]
                (ReqArg (\a o -> return o { optStoreNet = Just a }) "FILE")
                "write network to file"
          ]
