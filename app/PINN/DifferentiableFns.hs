{- Those language pragmas are needed to create and instantiate `DifferentiableFn` typeclass -}
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}

{-
    `PINN.DifferentiableFns` submodule provides `DifferentiableFn` typeclass which
    allows differentiation of functions. It needs programmer to manually derive
    the formula of a derivative of a function.
-}
module PINN.DifferentiableFns where

import Data.Vector(Vector)
import qualified Data.Vector as V

{-
    `DifferentiableFn` typeclass defines functions that take `args` and return `return_type`.

    `DifferentiableFn` must implement `call` and `derivative` functions.
-}
class DifferentiableFn fn args return_type where
    {-
        Calls function with supplied arguments.

        Since arrow operator is right-associative, you also may think that `call`
        takes a differentiable function and returns it as a plain Haskell function.
    -}
    call :: fn -> args -> return_type
    {-
        Calls derivative of a function with supplied arguments.

        Since arrow operator is right-associative, you also may think that `derivative`
        takes a differentiable function and returns its derivative as a plain Haskell function.
    -}
    derivative :: fn -> args -> return_type

{-
    `ActivationFn` enum lists activation functions that are supported.

    It is an instance of `DifferentiableFn` typeclass, which supplies functions to call and differentiate those functions.
-}
data ActivationFn = 
    {-
        `Id` function (also known as linear) is a function that does nothing with its argument.

        It is represented by `f(x) = x` and its derivative is `1`.
    -}
    Id |
    {-
        `ReLU` function (rectified linear unit) is equal to `Id` when the argument is positive, otherwise it is equal to 0.

        It is represented by `f(x) = max(0, x)` and its derivative is `1` when x >= 0 and `0` when x < 0.
    -}
    ReLU |
    {-
        `Sin` function is equal to sine of the angle in radians that is equal to the argument.

        It is represented by `f(x) = sin(x)` and its derivative is `cos(x)`.
    -}
    Sin
 deriving Show
instance DifferentiableFn ActivationFn Double Double where
    call fn = case fn of
                Id   -> id
                ReLU -> max 0
                Sin  -> sin
    
    derivative fn = case fn of
                        Id   -> const 1.0
                        ReLU -> fromIntegral . fromEnum . (>= 0)
                        Sin  -> cos
