{-
    `PINN.Model` submodule implements simple sequential neural network model,
    which is going to be used to solve Duffing equation. 
-}
module PINN.Model where

import Data.Vector (Vector)
import qualified Data.Vector as V

import Data.Matrix (Matrix)
import qualified Data.Matrix as M

{-
    `Layer` record datatype represents one layer of a neural network.
-}
data Layer = Layer {
    {-
        `weights` matrix represents weights of this layer.

        It must have the same number of rows as columns in the weight matrix of the previous layer,
        and the same number of columns as length of the bias vector.
    -}
    weights :: Matrix Double,
    {-
        `bias` vector represents bias of this layer.

        It must have the same length as number of columns of the `weights` matrix.
    -}
    bias :: Vector Double,

    {-
        `activation_fn` is an activation function of this layer.
    -}
    activation_fn :: Double -> Double
}

{-
    Returns the amount of neurons in a layer.
-}
layerSize :: Layer -> Int
layerSize = V.length . bias

{-
    `SequentialModel` record datatype represents simple sequential neural network
    where every layer is only connected with the previous.
-}
newtype SequentialModel = SequentialModel {
    {-
        `layers` is a vector of layers which this model consists of.
    -}
    layers :: Vector Layer
}

{-
    Returns the size of input values that model is compatible for.
-}
inputSize :: SequentialModel -> Int
inputSize = layerSize . V.head . layers
{-
    Returns the size of output values that model is compatible for.
-}
outputSize :: SequentialModel -> Int
outputSize = layerSize . V.last . layers

{-
    `assembleModel` function creates correct sequential model
    with layers of specified sizes and with specified activation functions. 

    'Correct sequential model' means that sizes of weights and biases of all layers are compatible
    (constrains that are described in the `weights` and `bias` functions of `Layer` datatype were satisfied).
-}
assembleModel :: [(Int, Double -> Double)] -> SequentialModel
assembleModel [] = SequentialModel V.empty
assembleModel ((size, fn):other_layers) = SequentialModel $ V.fromList (newLayer (1, size) fn : go size other_layers)
 where
    newLayer' :: ((Int, Int) -> Double, Int -> Double) -> (Int, Int) -> (Double -> Double) -> Layer
    newLayer' (weights_fn, bias_fn) (prev_size, layer_size) activation = layer
     where
        layer_weights = M.matrix prev_size layer_size weights_fn
        layer_bias = V.generate layer_size bias_fn
        layer = Layer layer_weights layer_bias activation
    
    newLayer :: (Int, Int) -> (Double -> Double) -> Layer
    newLayer = newLayer' (const 1.0, const 0.0)

    go :: Int -> [(Int, Double -> Double)] -> [Layer]
    go _ [] = []
    go prev_size ((layer_size, activation):xs) = newLayer (prev_size, layer_size) activation : go layer_size xs

{-
    `predict` function uses neural network to predict output values
    based on the state of the model and input values.

    `input` vector length must be equal to `inputSize` of model.
-}
predict :: SequentialModel -> Vector Double -> Vector Double
predict model input = M.getMatrixAsVector $ V.foldl' step (M.rowVector input) (layers model)
 where
    step :: Matrix Double -> Layer -> Matrix Double
    step prev layer = M.mapPos (const $ activation_fn layer) (prev * (weights layer) + (M.rowVector $ bias layer))
