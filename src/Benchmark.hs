{-# LANGUAGE CPP #-}
module Main (main) where

import Control.Concurrent (threadDelay)
import Control.Parallel.Strategies
import Data.List (foldl', intercalate)
import System.CPUTime
import System.FilePath
import System.IO
import System.IO.Unsafe (unsafeInterleaveIO)
import System.Random
import Test.QuickCheck
import Text.Printf

import Construction.Construction
import Construction.Izhikevich
import Construction.Network
import Construction.Neuron
import Construction.Synapse
import Examples.Ring (ring)
import Examples.Smallworld (smallworld, smallworldOrig)
import Examples.Random1k
import Options
import Simulation.Common
import Simulation.FiringStimulus
import Simulation.Run (initSim)
import Simulation.STDP (STDPApplication(..))
import Types
import qualified Util.List as L (chunksOf)


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
        headers = printCVSRow ["cycles", "elapsed", "fired", "spikes", "neurons", "synapses", "throughput", "speedup", "firing rate"]
        units   = printCVSRow [      "",    "(ms)",      "",       "",       "",          "", "(Mspikes/s)",       "",          "Hz"]

rtsCvs :: RTS -> String
rtsCvs (RunTimeStatistics cycles elapsed fired spikes neurons synapses) =
    printCVSRow $ map show [cycles, elapsed, fired, spikes, neurons, synapses, throughput] 
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
        -- firingRate = (realToFrac fired) / (realToFrac cycles)


-- RTS initialised with network statistics
initRTS :: Network (IzhNeuron FT) Static -> RTS
initRTS net = RunTimeStatistics {
        rtsCycles   = 0,
        rtsElapsed  = 0,
        rtsFired    = 0,
        rtsSpikes   = 0,
        -- TODO: change this back! (but evaluate much earlier)
        rtsNeurons  = 1000,
        -- rtsNeurons  = fromIntegral $! size net,
        rtsSynapses = 1000000
        -- rtsSynapses = fromIntegral $! length $! allSynapses $! neurons net
    }


data Benchmark = Benchmark {
        bmName     :: String,
        -- bmNet      :: Gen (Network (IzhNeuron FT) Static),
        bmNet      :: Network (IzhNeuron FT) Static,
        bmFStim    :: FiringStimulus,
        bmCycles   :: Int
    }



-- Scalable ring benchmark
{-
ringBenchmark n = Benchmark {
        bmName   = "ring",
        bmNet    = ring n 1,
        bmFStim  = FiringList [(0,[0])],
        bmCycles = 10000
    }


-- Scalable smallworld network
smBenchmark n = Benchmark {
        bmName   = "smallworld",
        bmNet    = smallworld (n `div` 1000) 1000 100 0.01 1 34 False,
        -- bmNet    = smallworld (n `div` 1000) 1000 100 0.01 1 29,
        bmFStim  = (FiringList $ map (\t -> (t, [1])) [0..8]),
        bmCycles = 10000
    }


smBenchmark' _ = Benchmark {
        bmName   = "smallworld",
        bmNet    = smallworldOrig,
        -- bmNet    = smallworld (n `div` 1000) 1000 100 0.01 1 29,
        bmFStim  = (FiringList $ map (\t -> (t, [1])) [0..8]),
        bmCycles = 10000
    }
-}

-- Local clusters network
localBenchmark n = Benchmark {
        bmName  = "local",
        -- bmNet   = smallworld (n `div` 1000) 1000 100 0.0 1 19,
        -- bmNet   = smallworld (n `div` 1000) 1000 100 0.0 1 34,
        -- ORIG bmNet   = smallworld n csize 100 0.0 1 34,
        -- bmNet   = smallworld n csize 100 0.0 5 3 True,
        --
        -- bmNet   = smallworld n csize 100 0.0 5 15 True,

        -- bmNet = random1k 820 204,
        -- bmNet = cluster (replicate n (random1k 820 204)) [],
        -- bmNet = localClusters n,
        bmNet = localClusters' 123456 n 1000,

        -- bmNet   = smallworld n csize 100 0.0 20 33,
        -- bmFStim = (FiringList $ map (\t -> (t, [1])) [0..8]),
        -- bmFStim = FiringList [(1, zipWith (replicate n fstim1))]
        -- bmFStim = FiringList [(1, allstim)],
        -- bmFStim = FiringList $ map (\t -> (t, allstim)) [0..1],
        bmFStim = NoFiring,
        -- bmFStim = FiringList [(1, take n $ iterate (+(n`div`20)) 0)],
        bmCycles = 10000
    }
    where
        -- fstim1 = take 20 $ iterate (+((n`div`20)-1)) 0
        -- fstim1 = take 20 $ iterate (+49) 0
        fstim1 = take 16 $ iterate (+50) 0
        ccount = n
        csize = 1024
        allstim = concatMap (\i -> map (+ (i*csize)) fstim1) [0..ccount-1]
        allstim' = concatMap (\i -> map (+ (i*csize)) [0..799] ) [0..ccount-1]



{-
-- Scalable uniform network
uniformBenchmark n = Benchmark {
        bmName  = "uniform",
        bmNet   = smallworld 1 n 100 0.0 1 34 False,
        -- bmNet   = smallworld 1 n 100 0.1 30,
        -- bmFStim = (FiringList $ map (\t -> (t, [1,50])) [0..12]),
        bmFStim = FiringList [(1, take 20 $ iterate (+(n`div`20)) 0)],
        -- bmFStim = FiringList [(1, [1..20])],
        -- bmFStim = FiringList [(1, [1])],
        bmCycles = 1000
    }
-}


-- For timing run we don't want to read back firing
#if defined(CUDA_ENABLED)
timingOptions = defaultOptions { optCuProbeDevice = False }
rtsOptions    = defaultOptions { optCuProbeDevice = True }
#else
timingOptions = defaultOptions
rtsOptions    = defaultOptions
#endif


{- Run several benchmarks and write data to CSV files -}
runBenchmarks :: Backend -> (Int -> Benchmark) -> [Int] -> IO ()
runBenchmarks _ bm [] = return ()
runBenchmarks backend bm sz = do
    -- TODO: clear the output files
    -- appendFile (filename name) rtsCvsHeader
    mapM_ (run backend bm) sz
    -- run backend bm 1
    where
        run backend bmF sz = do
            let bm = bmF sz
                -- name = bmName bm
-- {-
            -- putStrLn $ "Running benchmark " ++ name ++ "-" ++ show sz
            -- TODO: run in both backend modes at the same time, to avoid building network twice
            -- rts <- runBenchmark backend bm
            runBenchmark backend bm
            return ()
            -- appendFile (filename name) rtsCvsHeader
            -- appendFile (filename name) $ rtsCvs rts
            -- putStrLn $ rtsCvsHeader
            -- putStrLn $ rtsCvs rts
-- -}

filename series = "performance" </> (series ++ "-axel03") <.> "dat"


runBenchmark :: Backend -> Benchmark -> IO RTS
runBenchmark backend bm = do
    -- TODO: use RNG seed from time, and average over e.g. 3 runs
    let -- rngSeed = 123456
        -- net = build rngSeed $ bmNet bm
        -- net = bmNet bm
        net = bmNet bm `using` rnf
        rts0 = initRTS net
    rts1 <- runBenchmarkTiming backend net bm rts0
--    return rts1
-- {-
    -- let rts1 = rts0
    putStrLn $ show rts1
    rts2 <- runBenchmarkData backend net bm rts1
    -- rts2 <- runBenchmarkTiming backend net bm rts1
    -- putStrLn $ show rts2
    putStrLn rtsCvsHeader
    putStrLn $ rtsCvs rts2
    let throughput = (rtsSpikes rts2) * 1000 `div` (rtsElapsed rts2)
    putStrLn $ "Throughput: " ++ show (throughput `div` 1000000) ++ "M"
    return rts2
-- -}

runBenchmarkTiming backend net bm rts = do
    -- TODO: don't read data back!
    (Simulation sz run elapsed resetTimer close) <-
        initSim backend net All (Firing :: ProbeFn IzhState) 4 False timingOptions Nothing
    -- Note: only provide firing stimulus during the warm-up
    fstim <- firingStimulus $ bmFStim bm
    -- istim <- mapM (\_ -> sampleCurrentStimulus (bmIStim bm)) [0..initCycles-1]
    -- let stim = take initCycles $ map (\f -> (f, [], STDPIgnore)) fstim
    let stim = take initCycles $ map (\f -> (f, STDPIgnore)) fstim
    -- putStrLn $ show $ map (\(x, y, z) -> x) stim
    putStrLn "Warming up timing run"
    -- mapM_ (aux run) $ L.chunksOf sz stim
    -- mapM_ (aux run (\_ -> return ())) $ L.chunksOf sz stim
    mapM_ (step3 run) $ L.chunksOf sz stim
    putStrLn "Performing timing"
    -- t0 <- elapsed
    -- putStrLn $ "Elapsed 0: " ++ show t0
    -- putStrLn $ "waiting"
    -- threadDelay $ 3 * 1000000 
    resetTimer
    -- t1 <- elapsed
    -- putStrLn $ "Elapsed 1: " ++ show t1
    -- TODO: factor out timing code
    t0 <- getCPUTime
    let runCycles = bmCycles bm
        runStim = replicate (runCycles `div` sz) $ replicate sz ([], STDPIgnore)
    -- putStrLn $ "run stim: " ++ show (length runStim)
    mapM_ (step3 run) runStim
    t1 <- getCPUTime
    t <- elapsed
    putStrLn $ "Elapsed: " ++ show t
    putStrLn $ (show ((t1-t0) `div` 1000000000)) ++ " vs " ++ show t
    close
    return $ rts { rtsCycles = fromIntegral runCycles,
                   rtsElapsed = fromIntegral t }
                   -- rtsElapsed = (t1-t0) `div` 1000000000 }



{-
accFiringStats
    :: Neurons (IzhNeuron FT) Static
    -> ProbeData
    -> RTS
    -> RTS
accFiringStats _ (NeuronState _) rts = error "accFiringStats: non-firing data"
accFiringStats ns (FiringData fs) rts =
    let
        newFired  = fromIntegral $ length fs
        newSpikes = sum $ map (fromIntegral . length . synapsesOf ns) fs
    in rts {
        rtsFired  = (rtsFired rts) + newFired,
        rtsSpikes = (rtsSpikes rts) + newSpikes
    }
-}

initCycles = 1000

runBenchmarkData backend net bm rts = do
    (Simulation sz run elapsed resetTimer close) <-
        initSim backend net All (Firing :: ProbeFn IzhState) 4 False rtsOptions Nothing
    -- TODO: factor out stimulus
    -- Note: only provide firing stimulus during the warm-up
    fstim <- firingStimulus $ bmFStim bm
    let stim = take initCycles $ map (\f -> (f, STDPIgnore)) fstim
    putStrLn "Warming up data run"
    mapM_ (step3 run) $ L.chunksOf sz stim
    resetTimer
    putStrLn "Gathering data"
    -- let runStim = replicate (runCycles `div` sz) $ replicate sz ([],[], STDPIgnore)
    let runCycles = bmCycles bm
    let runStim = replicate (runCycles `div` sz) $ replicate sz ([], STDPIgnore)
    putStrLn $ "run stim: " ++ show (length runStim)
    -- TODO: get synapse count from benchmark
    rts' <- foldRTS (step3 run) (foldl' $ updateRTS) rts runStim
    close
    return rts'


firingCount :: ProbeData -> Integer
firingCount (FiringCount n) = fromIntegral n
firingCount (FiringData fs) = fromIntegral $ length fs
firingCount _               = error "firingCount: non-firing data"


{- | For speed: assume every synapse has the same number of synapes -}
-- updateRTS :: Integer -> RTS -> ProbeData -> RTS
updateRTS :: RTS -> ProbeData -> RTS
updateRTS rts p = rts {
        rtsFired = (rtsFired rts) + n,
        rtsSpikes = (rtsSpikes rts) + n * m
    }
    where
         n = firingCount p
         -- assume average distribution
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


{- Run simulation, but discard the results -}
foldSim_ :: (Monad m) => (b -> m a) -> [b] -> m ()
foldSim_ _ [] = return ()
foldSim_ step (x:xs) = do
    -- TODO: force evaulation here, if possible
    step x
    foldSim_ step xs


-- TODO: move this to general simulation code
{- | Run step function on packed stimulation data -}
-- TODO just use zipWithM
step3 run stim = do
    let (fstim, stdp) = unzip stim
    run fstim stdp


backend = CUDA
-- backend = CPU

-- main = runBenchmarks backend localBenchmark [1, 2] -- , 2, 4, 8, 16, 32]
main = runBenchmarks backend localBenchmark [2] -- , 8, 16, 32]
-- main = runBenchmarks backend smBenchmark' [1] --, 2, 4, 8, 16, 32]
-- main = runBenchmarks ringBenchmark $ [1000, 2000, 4000, 8000, 16000]
-- main = runBenchmarks backend smBenchmark $ [1000]
-- main = runBenchmarks smBenchmark $ [1000, 2000, 4000, 8000, 16000]
-- main = runBenchmarks smBenchmark $ [32000]
---main = runBenchmarks uniformBenchmark [1000, 2000, 4000]
-- main = runBenchmarks uniformBenchmark [4000, 8000, 16000]
-- main = runBenchmarks uniformBenchmark $ [32000]
-- main = runBenchmarks backend uniformBenchmark $ [1000]
