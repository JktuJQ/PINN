{-
    `CauchyProblem` module declares several basic types that
    are used to describe Cauchy boundary value problem.
-}
module CauchyProblem where

import Data.Vector (Vector)
import CauchyProblem.Times (Time)

{-
    `VarName` type alias refers to the name of variable in Cauchy problem.
-}
type VarName = String
{-
    `Vars` type alias stands for vector of variables.
-}
type Vars = Vector Double

{-
    `Parameters` type alias represents parameters of functions in Cauchy problem.
    It includes time parameter and vector of variables.    
-}
type Parameters = (Time, Vars)
{-
    `Fn` type alias represents some function `F(t, u)`, where `u` is a vector of (u_1, u_2, ..., u_n).
    
    In the Cauchy problem, this function represents
    the right side of the differential equation `d^k/dt^k(u_i) = F_i(t, u)`.
-}
type Fn = Parameters -> Double

{-
    `CauchyData` record datatype corresponds to Cauchy problem initial conditions.
-}
data CauchyData = CauchyData {
    {-
        `u0` is a vector of (u_0, u_1, ..., u_n) values at the `t0` time mark.
    -}
    u0 :: Vector Double,

    {-
        `fns` is a vector of functions (F_0, F_1, ..., F_n)
        that corresponds to the right side of `du_i/dt = F_i(t, u)` differential equation.
    -}
    fns :: Vector Fn
}
