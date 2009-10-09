module NemoFrontend_Client(setBackend,enableStdp,enablePipelining,pipelineLength,disableStdp,addNeuron,startSimulation,run,applyStdp,getConnectivity,stopSimulation,reset,terminate) where
import Data.IORef
import Thrift
import Data.Typeable ( Typeable )
import Control.Exception
import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Int
import Nemo_Types
import NemoFrontend
seqid = newIORef 0
setBackend (ip,op) arg_host = do
  send_setBackend op arg_host
  recv_setBackend ip
send_setBackend op arg_host = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("setBackend", M_CALL, seqn)
  write_SetBackend_args op (SetBackend_args{f_SetBackend_args_host=Just arg_host})
  writeMessageEnd op
  tFlush (getTransport op)
recv_setBackend ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_SetBackend_result ip
  readMessageEnd ip
  return ()
enableStdp (ip,op) arg_prefire arg_postfire arg_maxWeight = do
  send_enableStdp op arg_prefire arg_postfire arg_maxWeight
  recv_enableStdp ip
send_enableStdp op arg_prefire arg_postfire arg_maxWeight = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("enableStdp", M_CALL, seqn)
  write_EnableStdp_args op (EnableStdp_args{f_EnableStdp_args_prefire=Just arg_prefire,f_EnableStdp_args_postfire=Just arg_postfire,f_EnableStdp_args_maxWeight=Just arg_maxWeight})
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
disableStdp (ip,op) = do
  send_disableStdp op
  recv_disableStdp ip
send_disableStdp op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("disableStdp", M_CALL, seqn)
  write_DisableStdp_args op (DisableStdp_args{})
  writeMessageEnd op
  tFlush (getTransport op)
recv_disableStdp ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_DisableStdp_result ip
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
  recv_stopSimulation ip
send_stopSimulation op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("stopSimulation", M_CALL, seqn)
  write_StopSimulation_args op (StopSimulation_args{})
  writeMessageEnd op
  tFlush (getTransport op)
recv_stopSimulation ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_StopSimulation_result ip
  readMessageEnd ip
  return ()
reset (ip,op) = do
  send_reset op
  recv_reset ip
send_reset op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("reset", M_CALL, seqn)
  write_Reset_args op (Reset_args{})
  writeMessageEnd op
  tFlush (getTransport op)
recv_reset ip = do
  (fname, mtype, rseqid) <- readMessageBegin ip
  if mtype == M_EXCEPTION then do
    x <- readAppExn ip
    readMessageEnd ip
    throw x
    else return ()
  res <- read_Reset_result ip
  readMessageEnd ip
  return ()
terminate (ip,op) = do
  send_terminate op
send_terminate op = do
  seq <- seqid
  seqn <- readIORef seq
  writeMessageBegin op ("terminate", M_CALL, seqn)
  write_Terminate_args op (Terminate_args{})
  writeMessageEnd op
  tFlush (getTransport op)
