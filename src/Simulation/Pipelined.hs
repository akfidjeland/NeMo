{- | The backend simulation can be run in a separate thread. This enables us to
 - run simulation concurrently with other tasks such as communication with
 - other machines. -}

module Simulation.Pipelined (initSim) where

import Control.Concurrent
import Control.Concurrent.MVar

import Simulation (Simulation_Iface(..), Simulation(BS), Weights)
import qualified Simulation.Backend as Backend (initSim, SimulationOptions(optPipelined))
import Simulation.STDP (Reward)
import Types (Idx, FiringOutput(..))


{- We only access the simulation from inside the relevant thread. Although it
 - might be safe in some cases to access the simulation data from multiple
 - threads (perhaps controlled using MVar), this won't necessarily work in
 - general. In particular CUDA will fail if such accesses take place.
 -
 - We want to control the simulation using the regular interface
 - (Simulation_Iface). Because of the above constraints on data access we
 - unfortunately have to re-implement much of the interface in a way that's
 - accessable through thread synchronisation primitives. -}


data Command
    = CmdStep [Idx]  -- used for step and run
    | CmdPipelineLength
    | CmdApplyStdp Reward
    | CmdGetWeights
    | CmdStop


data Response
    = RspReady -- ... to start
    | RspDone  -- ... with termination
    | RspPipelineLength (Int, Int)
    | RspGetWeights Weights


{- In order to allow intermingling simulation commands (step, run, etc.) with
 - other commands query commands, we separate the output into firing output and
 - status. The firing channel can fill up during query command, if the input
 - pipeline needs to be flushed. -}
data SimulationThread = SimulationThread {
        input :: MVar Command,
        fired  :: Chan (Maybe FiringOutput),
        status :: MVar Response
    }


initSim net simOpts cudaOpts stdpConf = do
    if not (Backend.optPipelined simOpts)
        then Backend.initSim net simOpts cudaOpts stdpConf
        else do
            input  <- newEmptyMVar
            fired  <- newChan
            status <- newEmptyMVar
            let pl = SimulationThread input fired status
            forkOS $ do
                sim <- Backend.initSim net simOpts cudaOpts stdpConf
                start sim
                putMVar status RspReady -- signal that backend is ready ...
                writeChan fired Nothing -- ... but has no valid output data
                process pl sim
            takeMVar status
            return $! BS pl


{- | Process a command in simulation thread -}
process :: SimulationThread -> Simulation -> IO ()
process t sim = go =<< takeMVar (input t)
    where
        continueAfter f = f >> process t sim

        go (CmdStep fstim) = continueAfter $
            step sim fstim >>= writeChan (fired t) . Just

        go (CmdPipelineLength) = continueAfter $
            putMVar (status t) . RspPipelineLength =<< pipelineLength sim

        go (CmdApplyStdp reward) = continueAfter $ applyStdp sim reward

        go (CmdGetWeights) = continueAfter $
            putMVar (status t) . RspGetWeights =<< getWeights sim

        go CmdStop = stop sim >> putMVar (status t) RspDone



instance Simulation_Iface SimulationThread where

    {- Perform a complete non-pipelined simulation step.
     -
     - We provide no special 'run' function to accompany the 'step' function,
     - as we only have only buffer a single cycle's worth of firing at the
     - thread interface. The default method for run will thus suffice. We could
     - add a special run method with multiple cycle's of input and output by
     - using Concurrent.Chan, but it's unlikely that this will provide any
     - benefit -}
    step = threadStep

    {- When running in a pipeline step and step_ cannot be interspersed in any
     - sensible ways. Consider 'step >> step_'. The second function should
     - return the firing output from the first. However, the types simply do
     - not work out. -}
    step_ = fail "Pipelined simulation do not support discarding of output"

    pipelineLength = threadPipelineLength

    applyStdp = threadApplyStdp

    getWeights = threadGetWeights

    {- A 'start' function is not needed, as the simulation will have already
     - been set up when the thread was initially forked. -}

    stop = threadStop




{- | Send query on input channel, read data from status channel -}
threadQuery :: SimulationThread -> Command -> IO Response
threadQuery t cmd = putMVar (input t) cmd >> takeMVar (status t)


{- After step both firing input and output channels are full. -}
threadStep :: SimulationThread -> [Idx] -> IO FiringOutput
threadStep t fstim = do
    fired <- readChan (fired t)
    putMVar (input t) $ CmdStep fstim
    return $! maybe (FiringOutput []) id fired


threadPipelineLength :: SimulationThread -> IO (Int, Int)
threadPipelineLength t = do
    rsp <- threadQuery t CmdPipelineLength
    case rsp of
        RspPipelineLength (i, o) -> return $! (1+i, 1+o)
        _ -> fail "threadPipelineLength: unexpected response from simulation thread"


threadApplyStdp :: SimulationThread -> Reward -> IO ()
threadApplyStdp t reward = putMVar (input t) $ CmdApplyStdp reward


threadGetWeights :: SimulationThread -> IO Weights
threadGetWeights t = do
    rsp <- threadQuery t CmdGetWeights
    case rsp of
        RspGetWeights ws -> return ws
        _ -> fail "threadGetWeights: unexpected response from simulation thread"


{- We don't care about the result, but wait for it nonetheless to make sure
 - that the backend has been cleanly terminated. -}
threadStop :: SimulationThread -> IO ()
threadStop t = threadQuery t CmdStop >> return ()
