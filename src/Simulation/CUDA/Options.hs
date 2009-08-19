{- | Command-line options controlling CUDA backend -}

module Simulation.CUDA.Options (
        cudaOptions,
        optPartitionSize
    ) where

import Options


data CudaOptions = CudaOptions {
        optPartitionSize :: Maybe Int -- if nothing, use default in mapper
    }

cudaDefaults = CudaOptions {
        optPartitionSize = Nothing
    }

cudaOptions = OptionGroup "CUDA options" cudaDefaults cudaDescr

cudaDescr = [
        Option [] ["cuda-partition-size"]
            (ReqArg (\a o -> return o { optPartitionSize = Just $ read a }) "INT")
            -- (withDefault partitionSize "partition size for mapping onto CUDA MPs"),
            "partition size for mapping onto CUDA MPs"
    ]
