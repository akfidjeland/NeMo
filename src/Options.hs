{-# LANGUAGE CPP #-}

{- All options used by *any* of the programs. Individual programs may use only
 - a subset of these, or use some different default. -}

module Options (
    getOptions,
    -- * Option groups
    allOptions,
    defaultOptions,
    -- * Common options
    commonOptions,
    -- * Verbosity
    verbosityOptions,
        optVerbose,
    -- * Load/store options
    loadOptions,
        optLoadNet,
    storeOptions,
        optStoreNet,
    -- * Backend options
    serverBackendOptions,
    backendOptions,
        optBackend,
    -- * Simulation options
    simOptions,
        optDuration,
        optSeed,
        optTempSubres,
    stdpOptions,  -- extract all STDP options
    -- * Network options
    networkOptions,
        optPort,
    -- * CUDA-specific options
#if defined(CUDA_ENABLED)
    cudaOptions,
        optCuPartitionSz,
        optCuProbeDevice,
#endif
    -- * Print options
    printOptions,
        optDumpNeurons,
        optDumpMatrix
) where


import Data.Maybe
import System.Environment (getArgs)
import System.Console.GetOpt
import System.Exit
import System.IO (hPutStrLn, stderr)

import Types
import Simulation.Common
import Simulation.STDP
import Network.Protocol (defaultPort)


-- | Parse command line options
getOptions
    :: String               -- ^ name of binary (for help)
    -> Options              -- ^ default options
    -> [OptDescr (Options -> IO Options)]
    -> IO Options
getOptions progname defaultOptions options = do
    args <- getArgs
    let (actions, nonOpts, args', msgs) = getOpt' Permute options args
    if null msgs
        then do
            opt <- foldl (>>=) (return defaultOptions) actions
            if optShowHelp opt
                then showHelp progname options
                else return opt
        else do
            mapM (hPutStrLn stderr) msgs
            hPutStrLn stderr $
                "Run " ++ progname ++ " --help for summary of options"
            exitWith $ ExitFailure 1



data Options = Options {
        optShowHelp    :: Bool,
        -- verbosity
        optVerbose     :: Bool,
        -- load/store options
        optLoadNet     :: Maybe String,
        optStoreNet    :: Maybe String,
        -- simulation options
        optDuration    :: Duration,
        optSeed        :: Maybe Integer,
        optTempSubres  :: TemporalResolution,
        optStdpActive  :: Bool,
        optStdpFrequency :: Maybe Int,
        optStdpTauP    :: Int,
        optStdpTauD    :: Int,
        optStdpAlphaP  :: Double,
        optStdpAlphaD  :: Double,
        -- TODO: may want to use Maybe here, and default to max in network
        optStdpMaxWeight :: Double,
        optBackend     :: Backend,
        -- network options
        optPort        :: Int,
        -- print options
        optDumpNeurons :: Bool,
        optDumpMatrix  :: Bool,
        -- cuda backend options
        optCuPartitionSz :: Maybe Int,
        optCuProbeDevice :: Bool
    }


defaultOptions :: Options
defaultOptions = Options {
        optShowHelp    = False,
        -- verbosity
        optVerbose     = False,
        -- load/store options
        optStoreNet    = Nothing,
        optLoadNet     = Nothing,
        -- simulation options
        optDuration    = Forever,
        optSeed        = Nothing,
        optTempSubres  = 4,
        optStdpActive  = False,
        optStdpFrequency = Nothing,
        optStdpTauP    = 20,
        optStdpTauD    = 20,
        optStdpAlphaP  = 1.0,
        optStdpAlphaD  = 0.8,
        optStdpMaxWeight = 100.0, -- for no good reason at all...
        -- backend options
        optBackend     = defaultBackend,
        -- network options
        optPort        = defaultPort,
        -- print options
        optDumpNeurons = False,
        optDumpMatrix  = False,
        -- CUDA backend options
        optCuPartitionSz = Nothing,
        optCuProbeDevice = True
    }



-- TODO: add headers for each group when printing help
-- TODO: fill in the binary name here!
allOptions :: [OptDescr (Options -> IO Options)]
allOptions =
    commonOptions ++
    loadOptions ++
    storeOptions ++
    simOptions ++
    networkOptions ++
#if defined(CUDA_ENABLED)
    cudaOptions ++
#endif
    printOptions



-- Add default to option description
withDefault :: (Show a) => (Options -> a) -> String -> String
withDefault opt descr = descr ++ " (default: " ++ show (opt defaultOptions) ++ ")"



commonOptions = [
        Option ['h'] ["help"]
            (NoArg (\o -> return o { optShowHelp = True }))
            "show command-line options"
    ]


verbosityOptions :: [OptDescr (Options -> IO Options)]
verbosityOptions = [
        Option ['v'] ["verbose"]
            (NoArg (\o -> return o { optVerbose = True }))
            "more than usually verbose output"
    ]

storeOptions = [
        Option [] ["store-network"]
            (ReqArg (\a o -> return o { optStoreNet = Just a }) "FILE")
            "write network to file"
    ]

loadOptions = [
        Option [] ["load-network"]
            (ReqArg (\a o -> return o { optLoadNet = Just a }) "FILE")
            "load network from file"
    ]


serverBackendOptions :: [OptDescr (Options -> IO Options)]
serverBackendOptions = [
#if defined(CUDA_ENABLED)
        Option [] ["gpu"]
            (NoArg (\o -> return o { optBackend=CUDA }))
            (withDefault ((==CUDA) . optBackend) "use GPU backend for simulation, if present"),
#endif

        Option [] ["cpu"]
            (NoArg (\o -> return o { optBackend=CPU }))
            (withDefault ((==CPU) . optBackend) "use CPU backend for simulation")
    ]
#if defined(CUDA_ENABLED)
    ++ cudaOptions
#endif


backendOptions :: [OptDescr (Options -> IO Options)]
backendOptions = [
        Option [] ["remote"]
            (ReqArg getRemote "HOSTNAME[:PORT]")
            ("run simulation remotely on the specified server")
    ]
    ++ serverBackendOptions


-- format host:port
getRemote arg opts = return opts { optBackend = RemoteHost hostname port }
    where
        (hostname, port') = break (==':') arg
        port = if length port' > 1
            then read $ tail port'
            else defaultPort



simOptions = [

        Option ['t'] ["time"]    (ReqArg readDuration "INT")
            "duration of simulation in cycles (at 1ms resolution)",

        Option ['s'] ["seed"]    (ReqArg readSeed "INT")
            "seed for random number generation (default: system time)",

        Option [] ["temporal-subresolution"]
            (ReqArg (\a o -> return o { optTempSubres = read a }) "INT")
            (withDefault optTempSubres "number of substeps per normal time step"),

        Option [] ["stdp"]
            (NoArg (\o -> return o { optStdpActive = True }))
            "Enable STDP",

        Option [] ["stdp-frequency"]
            (ReqArg (\a o -> return o { optStdpFrequency = Just (read a) }) "INT")
             "frequency with which STDP should be applied (default: never)",

        Option [] ["stdp-a+"]
            (ReqArg (\a o -> return o { optStdpAlphaP = read a }) "FLOAT")
             (withDefault optStdpAlphaP "multiplier for synapse potentiation"),

        Option [] ["stdp-a-"]
            (ReqArg (\a o -> return o { optStdpAlphaD = read a }) "FLOAT")
             (withDefault optStdpAlphaD "multiplier for synapse depression"),

        Option [] ["stdp-t+"]
            (ReqArg (\a o -> return o { optStdpTauP = read a }) "INT")
             (withDefault optStdpTauP "Max temporal window for synapse potentiation"),

        Option [] ["stdp-t-"]
            (ReqArg (\a o -> return o { optStdpTauD = read a }) "INT")
             (withDefault optStdpTauD "Max temporal window for synapse depression"),

        Option [] ["stdp-max-weight"]
            (ReqArg (\a o -> return o { optStdpMaxWeight = read a }) "FLOAT")
             (withDefault optStdpMaxWeight "Set maximum weight for plastic synapses")
    ]


liftOpt6 c f1 f2 f3 f4 f5 f6 o = c (f1 o) (f2 o) (f3 o) (f4 o) (f5 o) (f6 o)

-- Extract STPD configuration from option structure
stdpOptions :: Options -> Maybe STDPConf
stdpOptions opts
    | optStdpActive opts = Just $ liftOpt6 STDPConf optStdpTauP optStdpTauD
                optStdpAlphaP optStdpAlphaD
                optStdpMaxWeight optStdpFrequency opts
    | otherwise = Nothing


printOptions = [

        Option ['C'] ["connectivity"]
            (NoArg (\o -> return o { optDumpMatrix=True }))
            "instead of simulating, just dump the connectivity matrix",

        Option ['N'] ["neurons"]
            (NoArg (\o -> return o { optDumpNeurons=True }))
            "instead of simulating, just dump the list of neurons"
    ]


networkOptions = [

        Option [] ["port"]
            (ReqArg (\a o -> return o { optPort = read a }) "INT")
            "port number for client/server communication"
    ]



#if defined(CUDA_ENABLED)
cudaOptions = [
        Option [] ["cuda-partition-size"]
            (ReqArg (\a o -> return o { optCuPartitionSz = Just $ read a }) "INT")
            (withDefault optCuPartitionSz "partition size for mapping onto CUDA MPs"),

        Option [] ["cuda-no-probe"]
            (NoArg (\o -> return o { optCuProbeDevice = False }))
            "don't read back probe data"
    ]
#endif

showHelp progname options = do
    putStr $ usageInfo ("\nusage: " ++ progname ++ " [OPTIONS]\n\nOptions:") options
    exitWith ExitSuccess


readDuration arg opt = return opt { optDuration = Until $ read arg }

readSeed arg opt = return opt { optSeed = Just $ read arg }
