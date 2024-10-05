{-
    `NumericalMethods.ErrorMargin` submodule implements functions
    that are useful in testing and calculating errors of numerical solutions.
-}
module CauchyProblem.NumericalMethods.ErrorTests where

import CauchyProblem.Times (Time)

{-
    Returns an infinite list of step functions where `step_i = step_{i-1} * k`.

    Since `TimeSettings` leaves an opportunity to use variable steps
    by requiring function `Int -> Time`,
    `iterateSteps` has to return `const step` to emulate function.
-}
iterateSteps :: Double -> Time -> [Int -> Time]
iterateSteps k start_step = map const (go start_step)
 where
    go step = step : go (step * k)

{-
    Calculates maximum absolute error between values of original function and numerical solution's one.
-}
absError :: (a -> a -> Double) -> (Time -> a) -> [(Time, a)] -> Double
absError diff original numerical_solution = maximum $ map (\(time, point) -> diff (original time) point) numerical_solution
