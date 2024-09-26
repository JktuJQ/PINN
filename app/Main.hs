module Main where

import CauchyProblem.Times (TimeSettings(..), timeline, fromTimeSettings)

main :: IO ()
main = do
    print $ timeline $ fromTimeSettings ( TimeSettings 0.0 10.0 (const 0.5) )
