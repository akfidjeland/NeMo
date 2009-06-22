{- | Command-line options controlling CUDA backend -}

module Simulation.CUDA.Options (
        cudaOptions,
        optPartitionSize,
        optProbeDevice
    ) where

import Options


data CudaOptions = CudaOptions {
        optPartitionSize :: Maybe Int, -- if nothing, use default in mapper
        -- TODO: move this to a SimulationOptions group
        optProbeDevice   :: Bool
    }

cudaDefaults = CudaOptions {
        optPartitionSize = Nothing,
        optProbeDevice   = True
    }

cudaOptions = OptionGroup "CUDA options" cudaDefaults cudaDescr

cudaDescr = [
        Option [] ["cuda-partition-size"]
            (ReqArg (\a o -> return o { optPartitionSize = Just $ read a }) "INT")
            -- (withDefault partitionSize "partition size for mapping onto CUDA MPs"),
            "partition size for mapping onto CUDA MPs",

        Option [] ["cuda-no-probe"]
            (NoArg (\o -> return o { optProbeDevice = False }))
            "don't read back probe data"
    ]
