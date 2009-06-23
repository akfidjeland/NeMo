{-# LANGUAGE CPP #-}
module Main (main) where

import Prelude

import Control.Monad (when)
import Control.Parallel.Strategies
import Data.List (foldl', intercalate)
import qualified Data.Map as Map (fromList)
import System.CPUTime (getCPUTime)
import System.Exit (exitWith, ExitCode(..))
import System.FilePath
import System.IO
import System.Random
import Text.Printf

import Construction hiding (excitatory, inhibitory, random)
import Construction.Neuron hiding (synapseCount)
import Options
import Simulation.Common
import Simulation.CUDA.Options (cudaOptions, optProbeDevice)
import Simulation.FiringStimulus
import Simulation.Options (simOptions, optBackend, SimulationOptions, BackendOptions(..))
import Simulation.Run (initSim)
import Simulation.STDP (STDPApplication(..), STDPConf(..), stdpOptions)
import Types
import qualified Util.List as L (chunksOf)


-------------------------------------------------------------------------------
-- Benchmark generation
-------------------------------------------------------------------------------

localClusters seed c m = Network ns t
    where
        -- random number generator which is threaded through the whole program
        r = mkStdGen seed
        sz = 1024
        n = c * sz
        ns = Map.fromList $ take n $ rneurons 0 m r
        -- the topology is not used after construction
        t = Node 0

-- Produce an infinite list of neurons
rneurons idx m r = (neuron' idx m r1) : (rneurons (idx+1) m r2)
    where
        (r1, r2) = split r

neuron' idx m r =
    if isExcitatory idx
        then (idx, excitatory idx m r)
        else (idx, inhibitory idx m r)
    where
        -- TODO: remove hard-coding here
        isExcitatory idx = idx `mod` 1024 < 820


-- excitatory neuron
exN r = mkNeuron2 0.02 b (v + 15*r^2) (8.0-6.0*r^2) u v thalamic
    where
        b = 0.2
        u = b * v
        v = -65.0
        thalamic = mkThalamic 5.0 r

excitatory pre m r = n `seq` neuron n ss
    where
        (nr, r2) = random r
        n = exN nr
        base = pre - (pre `mod` 1024)
        ss = exSS pre base m r2

-- exSS pre base r2 = zipWith (exSynapse pre) [base..base+1023] $ randoms r2
exSS pre base m r2 = ss `using` rnf
    where
        -- ss = map (exSynapse pre) [base..base+1023]
        ss = map (exSynapse pre) $ take m $ randomRs (base, base+1023) r2

-- TODO: randomise delays here!
exSynapse src tgt = Synapse src tgt 1 $! Static 0.25
-- exSynapse src tgt r = w `seq` StdSynapse src tgt w 1
    -- where w = 0.5 * r


-- inhibitory neuron
inN r = mkNeuron2 (0.02 + 0.08*r) b c 2.0 u v thalamic
    where
        b = 0.25 - 0.05 * r
        c = v
        u = b * v
        v = -65.0
        thalamic = mkThalamic 2.0 r

-- create a single inhibitory neuron based
-- inhibitory pre r = n `seq` (n, ss `using` rnf)
inhibitory pre m r = n `seq` neuron n ss
    where
        n = inN nr
        base = pre - (pre `mod` 1024)
        (nr, r2) = random r
        -- ss = zipWith (inSynapse pre) [base..base+1023] $ randoms r2
        -- ss = map (inSynapse pre) [base..base+1023]
        ss = map (inSynapse pre) $ take m $ randomRs (base, base+1023) r


inSynapse src tgt = Synapse src tgt 1 $ Static (-0.5)
-- inSynapse src tgt r = StdSynapse src tgt ((-1.0)*r) 1

-------------------------------------------------------------------------------
-- Run-time statistics
-------------------------------------------------------------------------------

data RTS = RunTimeStatistics {
        -- timing statistics
        rtsCycles    :: !Integer,   -- ^ number of ms simulation steps
        rtsElapsed   :: !Integer,   -- ^ elapsed real-time ms
        -- firing statisticcs
        rtsFired     :: !Integer,   -- ^ total number of neurons which fired
        rtsSpikes    :: !Integer,   -- ^ total number of spikes which were delivered
        -- network statistics
        rtsNeurons   :: !Integer,
        rtsSynapses  :: !Integer
        -- TODO: bandwidth
    } deriving (Show)



printCVSRow xs = intercalate ", " $ map (printf "%12s") xs

rtsCvsHeader :: String
rtsCvsHeader = ('#' : tail headers) ++ "\n" ++ ('#' : tail units)
    where
        headers = printCVSRow ["name", "cycles", "elapsed", "fired", "spikes", "neurons", "synapses", "throughput", "speedup", "firing rate"]
        units   = printCVSRow [    "",       "",    "(ms)",      "",       "",       "",          "", "(Mspikes/s)",       "",          "Hz"]


rtsCvs :: String -> RTS -> String
rtsCvs name (RunTimeStatistics cycles elapsed fired spikes neurons synapses) =
    printCVSRow $ [show name] ++ map show [cycles, elapsed, fired, spikes, neurons, synapses, throughput]
        ++ map (printf "%.3f") [speedup, firingRate]
    where
        throughput
            | elapsed == 0 = 0
            | otherwise    = spikes * 1000 `div` elapsed

        speedup :: Double
        speedup
            | elapsed == 0 = 0
            | otherwise    = (realToFrac cycles) / (realToFrac elapsed)

        firingRate :: Double
        firingRate = (realToFrac (fired*1000)) / (realToFrac (neurons*cycles))


{- RTS initialised with network statistics -}
initRTS :: Network (IzhNeuron FT) Static -> RTS
initRTS net = RunTimeStatistics {
        rtsCycles   = 0,
        rtsElapsed  = 0,
        rtsFired    = 0,
        rtsSpikes   = 0,
        rtsNeurons  = fromIntegral $! size net,
        rtsSynapses = fromIntegral $! synapseCount net
    }


data Benchmark = Benchmark {
        bmName     :: String,
        bmNet      :: Network (IzhNeuron FT) Static,
        bmFStim    :: FiringStimulus,
        bmCycles   :: Int
    }



-- Local clusters network
localBenchmark n m = Benchmark {
        bmName  = "local",
        bmNet = localClusters 123456 n m,
        bmFStim = NoFiring,
        bmCycles = 10000
    }



data Run = Timing | Data deriving (Eq)

-- For timing run we don't want to read back firing
cudaOpts run = (defaults cudaOptions) { optProbeDevice = (run == Data) }


runBenchmark :: SimulationOptions -> Benchmark -> IO ()
runBenchmark simOpts bm = do
    -- TODO: average over multiple runs
    let net = bmNet bm `using` rnf
        rts0 = initRTS net
    rts1 <- runBenchmarkTiming simOpts net bm rts0
    hPutStrLn stderr $ show rts1 -- intermediate results
    rts2 <- runBenchmarkData simOpts net bm rts1
    putStrLn $ rtsCvs (bmName bm) rts2
    let throughput = (rtsSpikes rts2) * 1000 `div` (rtsElapsed rts2)
    hPutStrLn stderr $ "Throughput: " ++ show (throughput `div` 1000000) ++ "M"

runBenchmarkTiming simOpts net bm rts = do
    -- TODO: don't read data back!
    (Simulation sz run elapsed resetTimer close) <-
        -- TODO: add STDP configuration
        initSim simOpts net All (Firing :: ProbeFn IzhState) False 
            (cudaOpts Timing) (defaults stdpOptions)
        -- initSim simOpts net All (Firing :: ProbeFn IzhState) False 
        --    (cudaOpts Timing) (defaults stdpOptions)
    -- Note: only provide firing stimulus during the warm-up
    fstim <- firingStimulus $ bmFStim bm
    let stim = take initCycles $ map (\f -> (f, STDPIgnore)) fstim
    hPutStrLn stderr "Warming up timing run"
    mapM_ (step3 run) $ L.chunksOf sz stim
    hPutStrLn stderr "Performing timing"
    resetTimer
    -- TODO: factor out timing code
    t0 <- getCPUTime
    let runCycles = bmCycles bm
        runStim = replicate (runCycles `div` sz) $ replicate sz ([], STDPIgnore)
    mapM_ (step3 run) runStim
    t1 <- getCPUTime
    t <- elapsed
    hPutStrLn stderr $ "Elapsed: " ++ show t
    hPutStrLn stderr $ (show ((t1-t0) `div` 1000000000)) ++ " vs " ++ show t
    close
    return $ rts { rtsCycles = fromIntegral runCycles,
                   rtsElapsed = fromIntegral t }


initCycles = 1000

runBenchmarkData simOpts net bm rts = do
    (Simulation sz run elapsed resetTimer close) <-
        initSim simOpts net All (Firing :: ProbeFn IzhState) False (cudaOpts Data) (defaults stdpOptions)
    -- TODO: factor out stimulus
    -- Note: only provide firing stimulus during the warm-up
    fstim <- firingStimulus $ bmFStim bm
    let stim = take initCycles $ map (\f -> (f, STDPIgnore)) fstim
    hPutStrLn stderr "Warming up data run"
    mapM_ (step3 run) $ L.chunksOf sz stim
    resetTimer
    hPutStrLn stderr "Gathering data"
    let runCycles = bmCycles bm
    let runStim = replicate (runCycles `div` sz) $ replicate sz ([], STDPIgnore)
    -- TODO: get synapse count from benchmark
    rts' <- foldRTS (step3 run) (foldl' $ updateRTS) rts runStim
    close
    return rts'


firingCount :: ProbeData -> Integer
firingCount (FiringCount n) = fromIntegral n
firingCount (FiringData fs) = fromIntegral $ length fs
firingCount _               = error "firingCount: non-firing data"


{- | For speed: assume every synapse has the same number of synapes -}
updateRTS :: RTS -> ProbeData -> RTS
updateRTS rts p = rts {
        rtsFired = (rtsFired rts) + n,
        rtsSpikes = (rtsSpikes rts) + n * m
    }
    where
         n = firingCount p
         -- assume uniform distribution
         m = (rtsSynapses rts) `div` (rtsNeurons rts)


-- b: stimulus
-- foldRTS :: (Monad m) => (b -> m a) -> (RTS -> a -> RTS) -> RTS -> [b] -> m RTS
foldRTS :: (b -> IO a) -> (RTS -> a -> RTS) -> RTS -> [b] -> IO RTS
foldRTS step update rts [] = return rts
foldRTS step update rts (x:xs) = do
    -- TODO: force evaulation here
    pdata <- step x
    let rts' = update rts pdata
    foldRTS step update rts' xs


-- TODO: move this to general simulation code
{- | Run step function on packed stimulation data -}
-- TODO just use zipWithM
step3 run stim = do
    let (fstim, stdp) = unzip stim
    run fstim stdp



data BenchmarkOptions = BenchmarkOptions {
        optN           :: Int,
        optM           :: Int,
        -- TODO: control cycles as well
        optBM          :: Int -> Int -> Benchmark,
        optPrintHeader :: Bool
    }

benchmarkDefaults = BenchmarkOptions {
        optN           = 1,
        optM           = 100,
        optBM          = localBenchmark,
        optPrintHeader = False
    }


createBenchmark :: BenchmarkOptions -> Benchmark
createBenchmark o = (optBM o) (optN o) (optM o)


benchmarkDescr = [
        Option ['M'] ["synapses"]
            (ReqArg (\a' o -> optRead "synapses" a' >>= \a -> return o{ optM = a }) "INT")
            "Number of synapses per neuron",

        Option ['n'] ["neurons"]
            (ReqArg (\a o -> return o { optN = read a }) "INT")
            "Number of neurons (thousands)",

        Option [] ["print-header"]
            (NoArg (\o -> return o { optPrintHeader=True }))
            "Print header for runtime statistics"
    ]

optRead :: Read a => String -> String -> Either String a
optRead optName s =
    case reads s of
        [(x, "")] -> Right x
        _         -> Left $ "Parse error of " ++ optName ++ ":" ++ s

benchmarkOptions = OptionGroup "Benchmark options" benchmarkDefaults benchmarkDescr





main = do
    (args, commonOpts) <- startOptProcessing
    simOpts <- processOptGroup (simOptions ServerBackends) args
    bmOpts  <- processOptGroup benchmarkOptions args
    endOptProcessing args
    when (optPrintHeader bmOpts) $ do
        putStrLn rtsCvsHeader
        exitWith ExitSuccess
    -- TODO: select benchmark from command-line
    -- TODO: control printing of header from the command-line
    runBenchmark simOpts (createBenchmark bmOpts)
