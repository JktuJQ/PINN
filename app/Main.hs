module Main where

import Data.Vector ((!))
import qualified Data.Vector as V

import CauchyProblem (CauchyData(..))
import CauchyProblem.Times (timeline, TimeSettings(..), fromTimeSettings)
import CauchyProblem.NumericalMethods (NumericalMethod)
import CauchyProblem.NumericalMethods.Methods (methodEuler, methodTrapezoid, methodRungeKutta)
import CauchyProblem.NumericalMethods.ErrorTests (absError)

test :: NumericalMethod -> Double
test method = absError (-) original_fn_y (zip ts y_solution)
 where
    x_eq (_, vars) = 2.0 * (vars!0) + (vars!1)
    y_eq (_, vars) = 3.0 * (vars!0) + 4.0 * (vars!1)

    timegrid = fromTimeSettings $ TimeSettings 0 1 (const 0.1)
    ts = timeline timegrid
    cauchy_data = CauchyData (V.fromList [1.0, 1.0]) (V.fromList [x_eq, y_eq])
    solution = method timegrid cauchy_data
    y_solution = V.toList $ V.map (! 1) solution

    original_fn_y t = (0.0 - 0.5 * exp t) + 1.5 * exp (5.0 * t)

main :: IO ()
main = do
    print $ test methodEuler
    print $ test methodTrapezoid
    print $ test methodRungeKutta
