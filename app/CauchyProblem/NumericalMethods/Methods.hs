{-
    `NumericalMethods.Methods` submodule provides implementations of classical numerical methods,
    such as Euler's explicit method, trapezoid method, Runge-Kutta method.
-}
module CauchyProblem.NumericalMethods.Methods where

import CauchyProblem (CauchyData(CauchyData))
import CauchyProblem.Times (TimeSettings(TimeSettings), Timegrid(Timegrid))
import CauchyProblem.NumericalMethods (NumericalMethod, iterStep, derivativeApprox)

{-
    Euler's explicit first-order numerical method.

    It is based on the simplest approximation of a derivative which is implemented
    as `derivativeApprox` in the `CauchyProblem.NumericalMethods` module.
-}
methodEuler :: NumericalMethod
methodEuler (Timegrid (TimeSettings _ _ step_fn) timeline) (CauchyData u0 fns) = (timeline, go (t, u0) ts)
 where
    approx i = derivativeApprox (step_fn i) fns
    go = iterStep approx

    t = head timeline
    ts = tail timeline
