module Benchmark (runBenchmark) where

import Prelude

import Control.Monad (when, foldM)
import Control.Parallel.Strategies
import Data.List (foldl', intercalate)
import System.CPUTime (getCPUTime)
import System.Exit (exitWith, ExitCode(..))
import System.IO
import System.Random
import Text.Printf

import Construction hiding (excitatory, inhibitory, random)
import Construction.Neuron
import qualified Construction.Neurons as Neurons (fromList)
import Options
import Simulation
import Simulation.CUDA.Options (cudaOptions)
import Simulation.FiringStimulus
import Simulation.Options (simOptions, optBackend, SimulationOptions, BackendOptions(..))
import Simulation.Backend (initSim)
import Simulation.STDP (StdpConf(..))
import Simulation.STDP.Options (stdpOptions)
import Types
import qualified Util.List as L (chunksOf)


-------------------------------------------------------------------------------
-- Benchmark generation: common
-------------------------------------------------------------------------------

-- TODO: remove hard-coding here
isExcitatory idx = idx `mod` 1024 < 820

exSynapse r tgt = AxonTerminal tgt delay 0.25 True ()
    where
        -- delay = 1
        delay = ceiling $ 20.0 * r
-- exSynapse r tgt = w `seq` AxonTerminal tgt 1 w ()
--    where w = 0.5 * r

-- TODO: randomise weights here!
inSynapse _ tgt = AxonTerminal tgt 1 (-0.5) False ()
-- inSynapse r tgt = AxonTerminal tgt 1 ((-1.0)*r) ()

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
    :: (FT -> IzhNeuron)       -- ^ function that generates neuron
    -> (FT -> Target -> AxonTerminal ())
                               -- ^ function that generates synapse
    -> Idx                     -- ^ presynaptic index
    -> Int                     -- ^ cluster count
    -> Int                     -- ^ cluster size
    -> Int                     -- ^ synapses per neuron
    -> Float                   -- ^ probability of local connection
    -> StdGen                  -- ^ random number generator
    -> Neuron IzhNeuron ()
clusteredNeuron' ngen sgen pre cc cs m p r = state `seq` neuron state ss
    where
        (nr, r2) = random r
        (r3, r4) = split r2
        sr = randoms r3
        state = ngen nr
        ss = zipWith sgen sr $ clusteredTargets pre cc cs m p r4


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
initRTS :: Network IzhNeuron () -> RTS
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
        bmNet      :: Network IzhNeuron (),
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



cudaOpts = defaults cudaOptions


{- Measure latency by always providing some stimulus and always reading back
 - data -}
runBenchmarkLatency :: SimulationOptions -> StdpConf -> Benchmark -> IO ()
runBenchmarkLatency simOpts stdpOpts bm = do
    putStrLn "Setting up sim"
    sim <- initSim net simOpts cudaOpts stdpOpts
    putStrLn "Warming up"
    run_ sim $ fstim 1000
    resetTimer sim
    putStrLn "Running"
    firing <- mapM (\f -> step sim f) $ fstim runCycles
    -- need to force evaluation of firing
    t <- rnf firing `seq` elapsed sim
    stop sim
    let rts1 = rts0 {
        rtsCycles = fromIntegral runCycles,
        rtsElapsed = fromIntegral t
    }
    let rts2 = foldl' updateRTS rts1 firing
    putStrLn $ rtsCvs (bmName bm) rts2
    let latency = (1000 * rtsElapsed rts2) `div` (rtsCycles rts2)
    hPutStrLn stderr $ "Latency: " ++ show latency ++ "ms/kcycle"
    where
        rts0 = initRTS net
        net = bmNet bm `using` rnf
        sz = size net
        runCycles = bmCycles bm
        {- We want /some/ stimulus, so that we do a copy operation. The amount of
         - data is not so important, so just stimulate a single neuron -}
        fstim n = map (\x -> [x `mod` sz]) [0..n-1]
        -- fstim n = map (\_ -> []) [0..n-1]


runBenchmarkThroughput :: SimulationOptions -> StdpConf -> Benchmark -> IO ()
runBenchmarkThroughput simOpts stdpOpts bm = do
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
    sim <- initSim net simOpts cudaOpts stdpOpts
    -- Note: only provide firing stimulus during the warm-up
    fstim <- firingStimulus $ bmFStim bm
    hPutStrLn stderr "Warming up timing run"
    run sim $ take initCycles fstim
    hPutStrLn stderr "Performing timing"
    resetTimer sim
    -- TODO: factor out timing code
    t0 <- getCPUTime
    run_ sim $ replicate runCycles []
    t1 <- getCPUTime
    t <- elapsed sim
    hPutStrLn stderr $ "Elapsed: " ++ show t
    hPutStrLn stderr $ (show ((t1-t0) `div` 1000000000)) ++ " vs " ++ show t
    stop sim
    return $ rts { rtsCycles = fromIntegral runCycles,
                   rtsElapsed = fromIntegral t }
    where
        chunkSize = maybe 1000 id (stdpFrequency stdpOpts)
        runCycles = bmCycles bm


initCycles = 1000

runBenchmarkData simOpts stdpOpts net bm rts = do
    sim <- initSim net simOpts cudaOpts stdpOpts
    -- Note: only provide firing stimulus during the warm-up
    fstim <- firingStimulus $ bmFStim bm
    hPutStrLn stderr "Warming up data run"
    run_ sim $ take initCycles fstim
    -- TODO: not needed:
    resetTimer sim
    hPutStrLn stderr "Gathering data"
    rts' <- foldM (\x _ -> runChunk sim x) rts $ [1.. (runCycles `div` chunkSize)]
    stop sim
    return rts'
    where
        chunkSize = maybe 1000 id (stdpFrequency stdpOpts)
        runCycles = bmCycles bm

        runChunk :: Simulation -> RTS -> IO RTS
        runChunk sim rts = do
            pdata <- run sim $ replicate chunkSize []
            return $! foldl' updateRTS rts pdata



{- | For speed: assume every synapse has the same number of synapes -}
updateRTS :: RTS -> FiringOutput -> RTS
updateRTS rts (FiringOutput f) = rts {
        rtsFired = (rtsFired rts) + n,
        rtsSpikes = (rtsSpikes rts) + n * m
    }
    where
         n = fromIntegral $ length f
         -- assume uniform distribution
         m = (rtsSynapses rts) `div` (rtsNeurons rts)




data BenchmarkOptions = BenchmarkOptions {
        optN           :: Int,
        optM           :: Int,
        optP           :: Float,
        -- TODO: control cycles as well
        optPrintHeader :: Bool,
        optMeasurement :: Measurement
    }

benchmarkDefaults = BenchmarkOptions {
        optN           = 1,
        optM           = 1000,
        optP           = 0.9,
        optPrintHeader = False,
        optMeasurement = Throughput
    }


data Measurement = Throughput | Latency

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
            "Print header for runtime statistics",

        Option [] ["throughput"]
            (NoArg (\o -> return o { optMeasurement = Throughput } ))
            "Measure throughput of system",

        Option [] ["latency"]
            (NoArg (\o -> return o { optMeasurement = Latency } ))
            "Measure latency of system"
    ]

optRead :: Read a => String -> String -> Either String a
optRead optName s =
    case reads s of
        [(x, "")] -> Right x
        _         -> Left $ "Parse error of " ++ optName ++ ":" ++ s


benchmarkOptions = OptionGroup "Benchmark options" benchmarkDefaults benchmarkDescr


runBenchmark args0 = do
    (args, commonOpts) <- startOptProcessing args0
    simOpts <- processOptGroup (simOptions AllBackends) args
    bmOpts  <- processOptGroup benchmarkOptions args
    stdpOpts<- processOptGroup stdpOptions args
    endOptProcessing args
    when (optPrintHeader bmOpts) $ do
        putStrLn rtsCvsHeader
        exitWith ExitSuccess
    let f = runFunction $ optMeasurement bmOpts
    f simOpts stdpOpts (createBenchmark bmOpts)
    where
        runFunction Throughput = runBenchmarkThroughput
        runFunction Latency = runBenchmarkLatency
