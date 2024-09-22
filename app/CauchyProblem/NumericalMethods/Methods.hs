{-
    `NumericalMethods.Methods` submodule provides implementations of classical numerical methods,
    such as Euler's explicit method, trapezoid method, Runge-Kutta method.
-}
module CauchyProblem.NumericalMethods.Methods where

import qualified Data.Vector as V

import CauchyProblem (CauchyData(CauchyData))
import CauchyProblem.Times (Timegrid(Timegrid))
import CauchyProblem.NumericalMethods (NumericalMethod, iterStep, derivativeApprox)

{-
    Euler's explicit first-order numerical method.

    It is based on the simplest approximation of a derivative which is implemented
    as `derivativeApprox` in the `CauchyProblem.NumericalMethods` module.
-}
methodEuler :: NumericalMethod
methodEuler (Timegrid tau_fn timeline) (CauchyData u0 fns) = (timeline, go (current_t, u0) ts)
 where
    current_t = head timeline
    ts = tail timeline

    approx i = derivativeApprox (tau_fn i) fns
    go = iterStep approx

{-
    'Trapezoidal rule' implicit second-order numerical method.

    It is based on the approximation of an integral with the trapezoid.
-}
methodTrapezoid :: NumericalMethod
methodTrapezoid (Timegrid tau_fn timeline) (CauchyData u0 fns) = (timeline, go (current_t, u0) ts)
 where
    current_t = head timeline
    ts = tail timeline

    approx i = derivativeApprox (tau_fn i) fns
    step i (t, u) = V.zipWith (\u_val f -> u_val + half_step * (f (t, u) + f middle_parameters)) u fns
     where
        tau = tau_fn i
        half_step = tau / 2.0
        recalculated = (approx i) (t, u)
        middle_parameters = (t + tau, recalculated)
    go = iterStep step

{-
    Runge-Kutta implicit fourth-order numerical method.

    It is based on the weighted approximation of slopes between two points,
    which allows for greater accuracy.
-}
methodRungeKutta :: NumericalMethod
methodRungeKutta (Timegrid tau_fn timeline) (CauchyData u0 fns) = (timeline, go (current_t, u0) ts)
 where
    current_t = head timeline
    ts = tail timeline
    
    step i (t, u) = u `add` ((tau / 6.0) `mul` k)
     where
        tau = tau_fn i
        
        add = V.zipWith (+)
        mul n = V.map (*n)

        k1 = V.map (\f -> f (t, u)) fns
        k2 = V.map (\f -> f (new_t, new_u)) fns
         where
            new_t = t + tau / 2.0
            new_u = u `add` ((tau / 2.0) `mul` k1)
        k3 = V.map (\f -> f (new_t, new_u)) fns
         where
            new_t = t + tau / 2.0
            new_u = u `add` ((tau / 2.0) `mul` k2)
        k4 = V.map (\f -> f (new_t, new_u)) fns
         where
            new_t = t + tau
            new_u = u `add` (tau `mul` k3)

        k = k1 `add` (2.0 `mul` k2) `add` (2.0 `mul` k3) `add` k4

    go = iterStep step
