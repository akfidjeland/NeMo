module Simulation.FiringStimulus (
    FiringStimulus(..),
    firingStimulus,
    denseToSparse
) where


import Types (Time, Idx)


-- should be sorted by time
type SparseFiring = [(Time, [Idx])]
type DenseFiring = [[Idx]]


-- | Return sparse firing stimulus
denseToSparse :: [[Idx]] -> [(Time, [Idx])]
denseToSparse firing = filter (not . null . snd) $ zip [0..] firing


-- We use one representation which is exposed to user code ...
data FiringStimulus
        = NoFiring
        | FiringFn (Time -> IO [Idx])
        | FiringList SparseFiring


-- Return lazy list of firing stimulus
firingStimulus :: FiringStimulus -> IO [[Idx]]
firingStimulus NoFiring = return $ repeat []
firingStimulus (FiringFn f) = mapM f [0..]
firingStimulus (FiringList fs) = return $ densify 0 fs
    where
        densify _ [] = repeat []
        densify t1 ((t2,x):xs) = replicate (t2-t1) [] ++ x : densify (t1+1) xs


instance Show FiringStimulus where
    show NoFiring = "NoFiring"
    show (FiringFn _) = "FiringFn"
    show (FiringList xs) = "FiringList" ++ show xs


-- Testing
-- TODO: implement generator
-- firing data should have unique time indices and should be sorted
-- generate a list, remove duplicats, order it
-- genSparseFiring :: Gen SparseFiring
-- genSparseFiring = sized $ \len -> vector len genFiring >>= liftM (sortBy fst)
