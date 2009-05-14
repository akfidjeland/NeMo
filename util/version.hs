#!/usr/bin/env runhaskell

import Data.Version
import Distribution.Package
import Distribution.PackageDescription
import Distribution.PackageDescription.Parse
import Distribution.Verbosity

main = do
    desc <- readPackageDescription silent "nemo.cabal"
    putStrLn $ showVersion $ pkgVersion $ package $ packageDescription desc
