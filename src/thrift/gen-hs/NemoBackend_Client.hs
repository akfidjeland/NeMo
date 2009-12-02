module NemoBackend_Client(addCluster,addNeuron,enableStdp,enablePipelining,pipelineLength,startSimulation,run,applyStdp,getConnectivity,stopSimulation) where
import Data.IORef
import Thrift
import Data.Typeable ( Typeable )
import Control.Exception
import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Int
import Nemo_Types
import NemoBackend
seqid = newIORef 0
addCluster (ip,op) arg_cluster = do
  send_addCluster op arg_cluster
  recv_addCluster ip
send_addCluster op arg_cluster = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("addCluster", M_CALL, seqn)
  write_AddCluster_args op (AddCluster_args{f_AddCluster_args_cluster=Just arg_cluster})
  writeMessageEnd op
  tFlush (getTransport op)
recv_addCluster ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_AddCluster_result ip
  readMessageEnd ip
  return ()
addNeuron (ip,op) arg_neuron = do
  send_addNeuron op arg_neuron
  recv_addNeuron ip
send_addNeuron op arg_neuron = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("addNeuron", M_CALL, seqn)
  write_AddNeuron_args op (AddNeuron_args{f_AddNeuron_args_neuron=Just arg_neuron})
  writeMessageEnd op
  tFlush (getTransport op)
recv_addNeuron ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_AddNeuron_result ip
  readMessageEnd ip
  case f_AddNeuron_result_err res of
    Nothing -> return ()
    Just _v -> throw _v
  return ()
enableStdp (ip,op) arg_prefire arg_postfire arg_maxWeight arg_minWeight = do
  send_enableStdp op arg_prefire arg_postfire arg_maxWeight arg_minWeight
  recv_enableStdp ip
send_enableStdp op arg_prefire arg_postfire arg_maxWeight arg_minWeight = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("enableStdp", M_CALL, seqn)
  write_EnableStdp_args op (EnableStdp_args{f_EnableStdp_args_prefire=Just arg_prefire,f_EnableStdp_args_postfire=Just arg_postfire,f_EnableStdp_args_maxWeight=Just arg_maxWeight,f_EnableStdp_args_minWeight=Just arg_minWeight})
  writeMessageEnd op
  tFlush (getTransport op)
recv_enableStdp ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_EnableStdp_result ip
  readMessageEnd ip
  case f_EnableStdp_result_err res of
    Nothing -> return ()
    Just _v -> throw _v
  return ()
enablePipelining (ip,op) = do
  send_enablePipelining op
  recv_enablePipelining ip
send_enablePipelining op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("enablePipelining", M_CALL, seqn)
  write_EnablePipelining_args op (EnablePipelining_args{})
  writeMessageEnd op
  tFlush (getTransport op)
recv_enablePipelining ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_EnablePipelining_result ip
  readMessageEnd ip
  return ()
pipelineLength (ip,op) = do
  send_pipelineLength op
  recv_pipelineLength ip
send_pipelineLength op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("pipelineLength", M_CALL, seqn)
  write_PipelineLength_args op (PipelineLength_args{})
  writeMessageEnd op
  tFlush (getTransport op)
recv_pipelineLength ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_PipelineLength_result ip
  readMessageEnd ip
  case f_PipelineLength_result_success res of
    Just v -> return v
    Nothing -> do
      throw (AppExn AE_MISSING_RESULT "pipelineLength failed: unknown result")
startSimulation (ip,op) = do
  send_startSimulation op
  recv_startSimulation ip
send_startSimulation op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("startSimulation", M_CALL, seqn)
  write_StartSimulation_args op (StartSimulation_args{})
  writeMessageEnd op
  tFlush (getTransport op)
recv_startSimulation ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_StartSimulation_result ip
  readMessageEnd ip
  case f_StartSimulation_result_err res of
    Nothing -> return ()
    Just _v -> throw _v
  return ()
run (ip,op) arg_stim = do
  send_run op arg_stim
  recv_run ip
send_run op arg_stim = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("run", M_CALL, seqn)
  write_Run_args op (Run_args{f_Run_args_stim=Just arg_stim})
  writeMessageEnd op
  tFlush (getTransport op)
recv_run ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_Run_result ip
  readMessageEnd ip
  case f_Run_result_success res of
    Just v -> return v
    Nothing -> do
      case f_Run_result_err res of
        Nothing -> return ()
        Just _v -> throw _v
      throw (AppExn AE_MISSING_RESULT "run failed: unknown result")
applyStdp (ip,op) arg_reward = do
  send_applyStdp op arg_reward
  recv_applyStdp ip
send_applyStdp op arg_reward = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("applyStdp", M_CALL, seqn)
  write_ApplyStdp_args op (ApplyStdp_args{f_ApplyStdp_args_reward=Just arg_reward})
  writeMessageEnd op
  tFlush (getTransport op)
recv_applyStdp ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_ApplyStdp_result ip
  readMessageEnd ip
  case f_ApplyStdp_result_err res of
    Nothing -> return ()
    Just _v -> throw _v
  return ()
getConnectivity (ip,op) = do
  send_getConnectivity op
  recv_getConnectivity ip
send_getConnectivity op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("getConnectivity", M_CALL, seqn)
  write_GetConnectivity_args op (GetConnectivity_args{})
  writeMessageEnd op
  tFlush (getTransport op)
recv_getConnectivity ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_GetConnectivity_result ip
  readMessageEnd ip
  case f_GetConnectivity_result_success res of
    Just v -> return v
    Nothing -> do
      throw (AppExn AE_MISSING_RESULT "getConnectivity failed: unknown result")
stopSimulation (ip,op) = do
  send_stopSimulation op
send_stopSimulation op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("stopSimulation", M_CALL, seqn)
  write_StopSimulation_args op (StopSimulation_args{})
  writeMessageEnd op
  tFlush (getTransport op)
