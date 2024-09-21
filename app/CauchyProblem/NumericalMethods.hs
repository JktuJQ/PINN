{-
    `NumericalMethods` submodule provides definitions for classical numerical methods.
-}
module CauchyProblem.NumericalMethods where

import Data.Vector (Vector, (!))
import qualified Data.Vector as V

import CauchyProblem (Vars, Fn, Parameters, CauchyData)
import CauchyProblem.Times (Time, Timeline, Timegrid)

{-
    `NumericalSolution` type alias represents the result of numerical method,
    which should produce values of `u` vector at all time marks in a timegrid.

    `NumericalSolution` also includes `Timeline` which belonged to a timegrid on which
    iteration was performed.
-}
type NumericalSolution = (Timeline, Vector Vars)
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
iterStep :: InductiveMethod -> Parameters -> Timeline -> Vector Vars
iterStep method (t, u) timeline = V.fromList $ go method (t, u) timeline
 where
    go _ _ [] = []
    go f (current_t, current_u) (new_t:ts) = current_u : go f (new_t, f (current_t, current_u)) ts
