{- Those language pragmas are needed to create and instantiate `Optimiser` typeclass -}
{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}

{-
    `PINN.Optimiser` submodule implements stochastic gradient descent
    optimiser for the sequential PINN model. 
-}
module PINN.Optimiser where

import Data.Vector(Vector)
import qualified Data.Vector as V

import Data.Matrix(Matrix)
import qualified Data.Matrix as M

import PINN.Model (Layer(..), SequentialModel(..))

{-
    `Optimiser` typeclass defines optimiser - algorithm that is able
    to improve model by minimising the loss function.
-}
class Optimiser optimiser params where
    {-
        `initializeParams` function returns initial layer specific parameters
        which are compatible with the model.

        Optimiser might need some additional information that will be available through the whole training.
        `initializeParams` will be called once when the training is started, and those parameters
        will change on layers with each iteration.
    -}
    initializeParams :: optimiser -> SequentialModel -> Vector params
    {-
        `updateStep` function implements optimisation of the training, namely the correction of weights after
        every iteration.

        Function takes layer specific parameters of the optimiser, learning rate, gradients of weights and bias
        and the layer which needs to be updated. Returned tuple contains updated layer and changed parameters of this step.
    -}
    updateStep :: optimiser -> params -> Double -> (Matrix Double, Vector Double) -> Layer -> (Layer, params)

{-
    `SGD` optimiser implements simple gradient descent with momentum.
-}
data SGD = SGD {
    {-
        `momentum` is a coefficient that specifies how much will the
        velocity (layer specific parameter) affect weights and bias.

        The greater the momentum is, the more the velocity's effect.

        Negative value of momentum might cause unwanted behaviour - `momentum` should be greater or equal to zero.
    -}
    momentum :: Double,
    {-
        `nesterov_momentum` flag switches the update rule (how will weights and bias be corrected).

        Considering `velocity = momentum * velocity - learning_rate * gradient`,
        when `nesterov_momentum` is `False` - `weights = weights + velocity`,
        when `nesterov_momentum` is `True` - `weights = weights + momentum * velocity - learning_rate * gradient`
        (the same goes for bias).
    -}
    nesterov_momentum :: Bool
}
type SGDVelocities = (Matrix Double, Vector Double)
instance Optimiser SGD SGDVelocities where
    initializeParams _ model = V.map initializeWithOnes (layers model)
     where
        initializeWithOnes :: Layer -> (Matrix Double, Vector Double)
        initializeWithOnes (Layer w b _) = (M.matrix (M.nrows w) (M.ncols w) (const 1.0), V.replicate (V.length b) 1.0)
    
    updateStep (SGD beta nesterov) (vdw, vdb) lr (dw, db) (Layer w b act_fn) = (Layer new_w new_b act_fn, (new_vdw, new_vdb))
     where
        newVelocity :: (a -> a -> a) -> (Double -> a -> a) -> a -> a -> a
        newVelocity diff_fn mul_fn gradient velocity = diff_fn (mul_fn beta velocity) (mul_fn lr gradient)
        
        w_velocity :: Matrix Double -> Matrix Double
        w_velocity = newVelocity (-) (M.mapPos . const . (*)) dw
        b_velocity :: Vector Double -> Vector Double
        b_velocity = newVelocity (V.zipWith (-)) (V.map . (*)) db

        new_vdw = w_velocity vdw 
        new_vdb = b_velocity vdb

        rule :: (a -> a) -> a -> a
        rule velocity_fn velocity = if nesterov then velocity_fn velocity else velocity

        w_sum = M.elementwise (+) w
        b_sum = V.zipWith (+) b
        (new_w, new_b) = (w_sum (rule w_velocity new_vdw), b_sum (rule b_velocity new_vdb))
