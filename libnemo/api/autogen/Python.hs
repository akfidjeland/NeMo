module Python (generate) where

import Text.PrettyPrint
import System.IO
import Data.Maybe (maybe)
import Data.Char
import Network.URI (escapeURIString)

import API
import Common

-- TODO: generate CPP macros instead

{- The API is generated using boost::python. All we need to do here is generate
 - the docstring -}
-- TODO: could generate most methods here, just except overloaded methods.
generate :: [ApiModule] -> IO ()
generate ms =
    withFile "../python/docstrings.h" WriteMode $ \hdl -> do
    hPutStr hdl $ render $ vcat (map moduleDoc ms) <> text "\n"


{- Generate global static named docstrings for each method -}
moduleDoc :: ApiModule -> Doc
moduleDoc mdl = vcat $ map (functionDoc (name mdl)) $ mdl_functions mdl


{- TODO: perhaps use actual formatting characters here -}
functionDoc :: String -> ApiFunction -> Doc
functionDoc mname fn = text "#define" <+> macroName <+> docstring
    where
        macroName = underscoredUpper [mname, camelToUnderscore (name fn), "doc"]
        docstring = doubleQuotes $ mergeLinesWith "\\n\\n" $ empty : filter (not . isEmpty) [synopsis, inputs, description]
        synopsis = text $ fn_brief fn
        inputs = inputDoc $ fn_inputs fn
        -- TODO: output doc
        description = maybe empty (text . escape) $ describe fn


inputDoc :: [Input] -> Doc
inputDoc [] = empty
inputDoc xs = mergeLines $ (text "Inputs:" : map (go . arg) xs)
    where
        -- TODO: deal with optional arguments here
        go :: ApiArg -> Doc
        go arg = text (name arg) <+> maybe empty (\a -> char '-' <+> text a) (describe arg)


-- functionName :: ApiFunction -> Doc
-- functionName = text . camelToUnderscore . name


-- qualifiedFunctionName :: String -> ApiFunction -> Doc
-- qualifiedFunctionName moduleName fn = text moduleName <> char '_' <> functionName fn


escape :: String -> String
escape xs = go xs
    where
        go [] = []
        go ('"':xs) = '\\' : '"' : go xs
        go (x:xs) = x : go xs