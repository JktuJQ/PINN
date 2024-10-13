{-
    `NumericalMethods` submodule provides definitions for classical numerical methods.
-}
module CauchyProblem.NumericalMethods where

import Data.Vector (Vector)
import qualified Data.Vector as V

import CauchyProblem (Vars, Fn, Parameters, CauchyData)
import CauchyProblem.Times (Time, Timeline, Timegrid)

{-
    `NumericalSolution` type alias represents the result of numerical method,
    which should produce values of `u` vector at all time marks in a timegrid.
-}
type NumericalSolution = Vector Vars
{-
    `NumericalMethod` type alias represents any numerical method that
    works on a Cauchy data and a specified timegrid and produces solution.
-}
type NumericalMethod = Timegrid -> CauchyData -> NumericalSolution

{-
    `InductiveMethod` represents a method that is able to produce new state
    from the old state.

    Iterative numerical methods are inductive, which means that they are able
    to calculate next value of a `u` vector using current values.
-}
type InductiveMethod = Parameters -> Vars

{-
    `iterStep` function inductively calculates result of iterative numerical method
    by iterating on a `Timeline`.
-}
iterStep :: (Int -> InductiveMethod) -> Parameters -> Timeline -> Vector Vars
iterStep method_creator (t, u) timeline = V.fromList $ go 0 method_creator (t, u) timeline
 where
    go :: Int -> (Int -> InductiveMethod) -> Parameters -> Timeline -> [Vars]
    go _ _ _ [] = []
    go i method (current_t, current_u) (new_t:ts) = current_u : go (i + 1) method new_parameters ts
     where
        new_parameters :: Parameters
        new_parameters = (new_t, method i (current_t, current_u))

{-
    Approximates derivative by its definition.

    `u' = lim ((u1 - u0) / tau) with tau -> 0`, so if you choose a small enough `tau`
    and you have functions for derivatives you can numerically approximate `u1 ≈ u0 + tau * u'`.
-}
derivativeApprox :: Time -> Vector Fn -> InductiveMethod
derivativeApprox tau fns = approx
 where
    approx :: InductiveMethod
    approx (t, u) = V.zipWith (\u_val f -> u_val + tau * f (t, u)) u fns
