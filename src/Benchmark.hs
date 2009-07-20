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
import qualified Construction.Neurons as Neurons (fromList)
import Options
import Simulation.Common
import Simulation.CUDA.Options (cudaOptions, optProbeDevice)
import Simulation.FiringStimulus
import Simulation.Options (simOptions, optBackend, SimulationOptions, BackendOptions(..))
import Simulation.Run (initSim)
import Simulation.STDP (STDPApplication(..), STDPConf(..))
import Simulation.STDP.Options (stdpOptions)
import Types
import qualified Util.List as L (chunksOf)


-------------------------------------------------------------------------------
-- Benchmark generation: common
-------------------------------------------------------------------------------

-- TODO: remove hard-coding here
isExcitatory idx = idx `mod` 1024 < 820

-- TODO: randomise delays here!
-- exSynapse r src tgt = Synapse src tgt 1 $! Static 0.25
exSynapse r src tgt = Synapse src tgt delay $! Static 0.25
    where
        delay = ceiling $ 20.0 * r
-- exSynapse src tgt r = w `seq` StdSynapse src tgt w 1
    -- where w = 0.5 * r

-- TODO: randomise weights here!
inSynapse _ src tgt = Synapse src tgt 1 $ Static (-0.5)
-- inSynapse src tgt r = StdSynapse src tgt ((-1.0)*r) 1

-- excitatory neuron
exN r = mkNeuron2 0.02 b (v + 15*r^2) (8.0-6.0*r^2) u v thalamic
    where
        b = 0.2
        u = b * v
        v = -65.0
        thalamic = mkThalamic 5.0 r


-- inhibitory neuron
inN r = mkNeuron2 (0.02 + 0.08*r) b c 2.0 u v thalamic
    where
        b = 0.25 - 0.05 * r
        c = v
        u = b * v
        v = -65.0
        thalamic = mkThalamic 2.0 r


-------------------------------------------------------------------------------
-- Benchmark: clustered
-------------------------------------------------------------------------------


clusteredNetwork seed cc m p = Network ns t
    where
        r  = mkStdGen seed -- thread RNG through whole program
        cs = 1024
        n  = cc * cs
        ns = Neurons.fromList $ take n $
                clusteredNeurons (exN, exSynapse) (inN, inSynapse) 0 cc cs m p r
        t  = Node 0 -- the topology is not used after construction


{- Return an infinite list of neurons which are clustered -}
clusteredNeurons ex_gen in_gen idx cc cs m p r = h : t
    where
        (r1, r2) = split r
        h = clusteredNeuron  ex_gen in_gen  idx    cc cs m p r1
        t = clusteredNeurons ex_gen in_gen (idx+1) cc cs m p r2


-- TODO: share with other generators
clusteredNeuron (ex_ngen, ex_sgen) (in_ngen, in_sgen) idx cc cs m p r =
    if isExcitatory idx
        then (idx, clusteredNeuron' ex_ngen ex_sgen idx cc cs m p r)
        else (idx, clusteredNeuron' in_ngen in_sgen idx cc cs m p r)
    where
        isExcitatory idx = idx `mod` cs < (cs * 8 `div` 10)


{- Generate a neuron with some local and some global connections. -}
clusteredNeuron'
    :: (FT -> IzhNeuron FT)    -- ^ function that generates neuron
    -> (FT -> Idx -> Idx -> Synapse Static)
                               -- ^ function that generates synapse
    -> Idx                     -- ^ presynaptic index
    -> Int                     -- ^ cluster count
    -> Int                     -- ^ cluster size
    -> Int                     -- ^ synapses per neuron
    -> Float                   -- ^ probability of local connection
    -> StdGen                  -- ^ random number generator
    -> Neuron (IzhNeuron FT) Static
clusteredNeuron' ngen sgen pre cc cs m p r = state `seq` neuron state ss
    where
        (nr, r2) = random r
        (sr, r3) = random r2
        state = ngen nr
        ss = map (sgen sr pre) $ clusteredTargets pre cc cs m p r3


{- Produce a list of postsynaptic neurons where a proportion 'p' is selected
 - from the local cluster (of the given size), whereas the rest are taken from
 - the whole neuron collection -}
-- clusteredTargets pre cc cs n m p r = l0 `seq` l1 `seq` l0 ++ l1
clusteredTargets pre cc cs m p r = l0 ++ l1
    where
        (r1, r2) = split r
        base = pre - (pre `mod` cs)
        l0_count = round $ (realToFrac m) * p
        l1_count = m - l0_count
        l0 = take l0_count $ randomRs (base, base+cs-1) r1
        {- The L1 conections should /not/ point to current cluster -}
        -- l1 = take l1_count $ randomRs (0, cc*cs-1) r2
        l1 = if cc > 1
                then take l1_count $ map adjust $ randomRs (0, (cc-1)*cs-1) r2
                else []
        adjust i = if i >= base then i+cs else i


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



clusteredBenchmark n m p = Benchmark {
        bmName = "clustered-" ++ show p,
        -- TODO: get seed from options
        bmNet = clusteredNetwork 123456 n m p,
        bmFStim = NoFiring,
        -- TODO: get cycles from options
        bmCycles = 10000
    }


data Run = Timing | Data deriving (Eq)

-- For timing run we don't want to read back firing
cudaOpts run = (defaults cudaOptions) { optProbeDevice = (run == Data) }


runBenchmark :: SimulationOptions -> STDPConf -> Benchmark -> IO ()
runBenchmark simOpts stdpOpts bm = do
    -- TODO: average over multiple runs
    let net = bmNet bm `using` rnf
        rts0 = initRTS net
    rts1 <- runBenchmarkTiming simOpts stdpOpts net bm rts0
    hPutStrLn stderr $ show rts1 -- intermediate results
    rts2 <- runBenchmarkData simOpts stdpOpts net bm rts1
    putStrLn $ rtsCvs (bmName bm) rts2
    let throughput = (rtsSpikes rts2) * 1000 `div` (rtsElapsed rts2)
    hPutStrLn stderr $ "Throughput: " ++ show (throughput `div` 1000000) ++ "M"

runBenchmarkTiming simOpts stdpOpts net bm rts = do
    (Simulation sz run elapsed resetTimer close) <-
        initSim simOpts net All (Firing :: ProbeFn IzhState) False 
            (cudaOpts Timing) stdpOpts
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

runBenchmarkData simOpts stdpOpts net bm rts = do
    (Simulation sz run elapsed resetTimer close) <-
        initSim simOpts net All (Firing :: ProbeFn IzhState) False 
            (cudaOpts Data) stdpOpts
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
        optP           :: Float,
        -- TODO: control cycles as well
        optPrintHeader :: Bool
    }

benchmarkDefaults = BenchmarkOptions {
        optN           = 1,
        optM           = 100,
        optP           = 0.9,
        optPrintHeader = False
    }


createBenchmark :: BenchmarkOptions -> Benchmark
createBenchmark o = clusteredBenchmark (optN o) (optM o) (optP o)


benchmarkDescr = [
        Option ['m'] ["synapses"]
            (ReqArg (\a' o -> optRead "synapses" a' >>= \a -> return o{ optM = a }) "INT")
            "Number of synapses per neuron",

        Option ['n'] ["neurons"]
            (ReqArg (\a o -> return o { optN = read a }) "INT")
            "Number of neurons (thousands)",

        Option ['p'] ["local-probability"]
            -- TODO: range-check
            (ReqArg (\a o -> return o { optP = read a }) "FLOAT")
            "Probability of local connections",

        Option ['H'] ["print-header"]
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
    simOpts <- processOptGroup (simOptions LocalBackends) args
    bmOpts  <- processOptGroup benchmarkOptions args
    stdpOpts<- processOptGroup stdpOptions args
    endOptProcessing args
    when (optPrintHeader bmOpts) $ do
        putStrLn rtsCvsHeader
        exitWith ExitSuccess
    runBenchmark simOpts stdpOpts (createBenchmark bmOpts)
