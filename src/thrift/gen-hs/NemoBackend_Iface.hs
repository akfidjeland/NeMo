module NemoBackend_Iface where
import Thrift
import Data.Typeable ( Typeable )
import Control.Exception
import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.Int
import Nemo_Types

class NemoBackend_Iface a where
  addCluster :: a -> Maybe [IzhNeuron] -> IO ()
  addNeuron :: a -> Maybe IzhNeuron -> IO ()
  enableStdp :: a -> Maybe [Double] -> Maybe [Double] -> Maybe Double -> IO ()
  enablePipelining :: a -> IO ()
  pipelineLength :: a -> IO PipelineLength
  startSimulation :: a -> IO ()
  run :: a -> Maybe [Stimulus] -> IO [[Int]]
  applyStdp :: a -> Maybe Double -> IO ()
  getConnectivity :: a -> IO (Map.Map Int [Synapse])
  stopSimulation :: a -> IO ()
