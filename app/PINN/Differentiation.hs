{- Those language pragmas are needed to create and instantiate `DifferentiableFn` typeclass -}
{-# LANGUAGE FlexibleInstances, FunctionalDependencies #-}

{-
    `PINN.Differentiation` submodule provides `DifferentiableFn` typeclass which
    allows differentiation of functions. It needs programmer to manually derive
    the formula of a derivative of a function.
-}
module PINN.Differentiation where

import Data.Vector (Vector)
import qualified Data.Vector as V

import Data.Hashable (Hashable(..))
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM

{-
    `Symbol` record datatype represents symbolic variable.
    It is needed to implement symbolic differentiation.

    `Symbol a` implements `Eq` and partly `Num` typeclasses if `a` implements them.
-}
data Symbol a = Symbol {
    {-
        Value of the symbolic variable.
    -}
    value :: a,
    {-
        List of gradients (with respect to operands).
    -}
    gradients :: [(Symbol a, Symbol a -> Symbol a)]
}

instance Eq a => Eq (Symbol a) where
    (==) a b = (value a) == (value b)
instance Hashable a => Hashable (Symbol a) where
    hashWithSalt salt (Symbol v grads) = salt `hashWithSalt` v `hashWithSalt` map fst grads

{-
    `Symbolic` typeclass defines types that are able to be treated symbolically.

    There is a blanket instance for `Symbol a` if `a` is `Symbolic`.
-}
class (Hashable a) => Symbolic a where
    {-
        Returns value that can be thought of as zero.
    -}
    constZero :: Symbol a
    {-
        Returns value that can be thought of as one.
    -}
    constOne :: Symbol a

    {-
        Creates new symbol from constant
    -}
    newSymbol :: a -> Symbol a
    newSymbol x = Symbol x []

    {-
        Symbolic addition.
    -}
    add :: Symbol a -> Symbol a -> Symbol a

    {-
        Symbolic multiplication.
    -}
    mul :: Symbol a -> Symbol a -> Symbol a

    {-
        Symbolic unary negation.
    -}
    neg :: Symbol a -> Symbol a
    neg x = constZero `sub` x
    {-
        Symbolic difference.
    -}
    sub :: Symbol a -> Symbol a -> Symbol a
    sub a b = a `add` (neg b)
instance Symbolic Double where
    constZero = Symbol 0.0 []
    constOne = Symbol 1.0 []

    add a b = Symbol (value a + value b) [(a, \path -> path), (b, \path -> path)]

    mul a b = Symbol (value a * value b) [(a, \path -> path `mul` b), (b, \path -> path `mul` a)]

    neg x = Symbol (-(value x)) [(x, \path -> neg path)]
    sub a b = Symbol (value a - value b) [(a, \path -> path), (b, \path -> neg path)]
 
{-
    `getGradients` function obtains the map of derivatives of given variable with respect to other variables.
-}
getGradients :: (Symbolic a) => Symbol a -> HashMap (Symbol a) (Symbol a)
getGradients variable = traverseGraph (gradients variable) (HM.singleton variable constOne) constOne
 where
    traverseGraph :: (Symbolic a) => [(Symbol a, Symbol a -> Symbol a)] -> HashMap (Symbol a) (Symbol a) -> Symbol a -> HashMap (Symbol a) (Symbol a)
    traverseGraph ((child, applyGradient):xs) grads path = traverseGraph xs down_grads path
     where
        child_path = applyGradient path
        
        new_grads = HM.alter (addPaths child_path) child grads
         where
            addPaths :: (Symbolic a) => Symbol a -> Maybe (Symbol a) -> Maybe (Symbol a)
            addPaths a (Just b) = Just (a `add` b)
            addPaths a Nothing = Just a

        down_grads = traverseGraph (gradients child) new_grads child_path
    traverseGraph [] grads _ = grads


{-
    `DifferentiableFn` typeclass defines functions that take `args` and return `return_type`.

    `DifferentiableFn` must implement `call` and `derivative` functions.
-}
class DifferentiableFn fn args return_type | fn -> args return_type where
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
    Sin |
    {-
        `Sigmoid` function is equal to `1 / (1 + exp (-x))`.

        It is represented by `f(x) = 1 / (1 + exp (-x))` and its derivative is `f(x) * (1 - f(x))`.
    -}
    Sigmoid |
    {-
        `Tanh` function is equal to `(exp (2 * x) - 1) / (exp (2 * x) + 1)`.

        It is represented by `f(x) = (exp (2 * x) - 1) / (exp (2 * x) + 1)` and its derivative is `1 - f(x) * f(x)`.
    -}
    Tanh
 deriving Show
instance DifferentiableFn ActivationFn Double Double where
    call fn x = case fn of
                    Id      -> id x
                    ReLU    -> max 0 x
                    Sin     -> sin x
                    Sigmoid -> 1.0 / (1.0 + exp (-x))
                    Tanh    -> (exp (2.0 * x) - 1.0) / (exp (2.0 * x) + 1.0)

    derivative fn x = case fn of
                        Id      -> 1.0
                        ReLU    -> fromIntegral $ fromEnum $ x >= 0
                        Sin     -> cos x
                        Sigmoid -> let s = call fn x in s * (1.0 - s)
                        Tanh    -> 1.0 - (call fn x) ^ (2 :: Int)

{-
    `LossFn` enum lists loss functions that are supported.

    It is an instance of `DifferentiableFn` typeclass, which supplies functions to call and differentiate those functions.
-}
data LossFn =
    {-
        `SSR` function (sum of squared residuals) is a function
        that computes the sum of squares of elements of the difference of two vectors.

        It is represented by `f(original, predicted) = sum(from i=1 to n) of (original_i - predicted_i)^2` where `n = length of original`.
    -}
    SSR |
    {-
        `MSE` function (mean squared error) is a function
        that computes the mean of the sum of squares of elements of the difference of two vectors.

        It is represented by `f(original, predicted) = SSR(original, predicted) / n` where `n = length of original`.
    -}
    MSE
instance DifferentiableFn LossFn (Vector Double, Vector Double) (Vector Double) where
    call fn args = case fn of
                       SSR -> V.map (^ (2 :: Int)) $ uncurry (V.zipWith (-)) args
                       MSE -> V.map (/ fromIntegral (V.length $ fst args)) $ call SSR args 

    derivative fn args = case fn of
                             SSR -> V.map ((0-) . (2*)) $ uncurry (V.zipWith (-)) args
                             MSE -> V.map (/ fromIntegral (V.length $ fst args)) $ derivative SSR args
