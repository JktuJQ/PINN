{-
    `NumericalMethods.Methods` submodule provides implementations of classical numerical methods,
    such as Euler's explicit method, trapezoid method, Runge-Kutta method.
-}
module CauchyProblem.NumericalMethods.Methods where

import Data.Vector ((!))
import qualified Data.Vector as V

import CauchyProblem (CauchyData(CauchyData))
import CauchyProblem.Times (TimeSettings(TimeSettings), Timegrid(Timegrid))
import CauchyProblem.NumericalMethods (NumericalMethod, iterStep, derivativeApprox)

{-
    Euler's explicit first-order numerical method.

    It is based on the simplest approximation of a derivative which is implemented
    as `derivativeApprox` in the `CauchyProblem.NumericalMethods` module.
-}
methodEuler :: NumericalMethod
methodEuler (Timegrid (TimeSettings _ _ step_fn) timeline) (CauchyData u0 fns) = (timeline, go (current_t, u0) ts)
 where
    current_t = head timeline
    ts = tail timeline

    approx i = derivativeApprox (step_fn i) fns
    go = iterStep approx

{-
    'Trapezoidal rule' implicit second-order numerical method.

    It is based on the approximation of an integral with the trapezoid.
-}
methodTrapezoid :: NumericalMethod
methodTrapezoid (Timegrid (TimeSettings _ _ step_fn) timeline) (CauchyData u0 fns) = (timeline, go (current_t, u0) ts)
 where
    current_t = head timeline
    ts = tail timeline

    approx i = derivativeApprox (step_fn i) fns
    step i (t, prev_u) = V.fromList [
                          (prev_u!index) + half_step
                          * ((fns!index) (t, prev_u) + ((fns!index) middle_parameters))
                          | index <- [0..(V.length prev_u)]
                         ]
     where
        half_step = (step_fn i) / 2.0
        recalculated = (approx i) (t, prev_u)
        middle_parameters = ((t + (step_fn i)), recalculated)
    go = iterStep step
