{-
    The goal of this project is to solve Duffing equation and
    plot Duffing oscillator using
    Physics-Informed Neural Network (PINN) and numerical methods.
    `Task` module defines this exact task.
-}
module Task where

import Data.Vector (Vector, (!))
import qualified Data.Vector as V

import CauchyProblem (Fn)

{-
    `x0` is a starting position of a Duffing oscillator.
-}
x0 :: Double -> String -> Double
x0 default_value input = if input == "" then default_value else read input
{-
    `x'0` is a starting velocity of a Duffing oscillator.
-}
x'0 :: Double -> String -> Double
x'0 default_value input = if input == "" then default_value else read input

{-
    `differentialEquations` function returns equations that define Duffing oscillator
     (dv/dt = gamma * cos (omega * t) - alpha * x - beta * x ^ 3 - delta * v).

    This function requires 5 parameters which can affect the oscillator.
-}
differentialEquations :: (Double, Double, Double, Double, Double) -> Vector Fn
differentialEquations (alpha, beta, gamma, delta, omega) = V.fromList [x_eq, x'_eq]
 where
    x_eq (_, vars) = vars!1
    x'_eq (t, vars) = gamma * cos (omega * t) - alpha * (vars!0) - beta * (vars!0) ^ (3 :: Int) - delta * (vars!1)
