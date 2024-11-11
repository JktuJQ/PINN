{-
    `PINN.Model` submodule implements simple sequential neural network model.
    
    It also provides `Layer` datatype that holds weights and bias of each single part of model.
-}
module PINN.Model where

import System.IO

import Data.Vector (Vector)
import qualified Data.Vector as V

import Data.Matrix (Matrix)
import qualified Data.Matrix as M

import PINN.DifferentiableFns (DifferentiableFn(call), ActivationFn)

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
    activation_fn :: ActivationFn
} deriving Show
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
} deriving Show
{-
    Returns the size of the input that the model is compatible for.

    The number `i` that this function returns means that this model
    is able to process matrices with sizes `(n, i)` where `n` is a number
    of different values. Docs on `process`, `process'` tell more about that.
-}
inputSize :: SequentialModel -> Int
inputSize = M.nrows . weights . V.head . layers
{-
    Returns the size of the output that the model is compatible for.

    The number `o` that this function returns means that this model's
    prediction returns matrices with sizes `(n, o)` where `n` is a number
    of different values. Docs on `process`, `process'` tell more about that.
-}
outputSize :: SequentialModel -> Int
outputSize = layerSize . V.last . layers

{-
    `WeightsInitFn` type alias describes function, with which
    you can initialize `weights` matrix for the `Layer`.

    First argument is the size of weights matrix, the second is the index in that matrix.

    You may want to initialize weights with constant value (`const $ const 1.0`) or
    with random values.
-}
type WeightsInitFn = (Int, Int) -> (Int, Int) -> Double
{-
    `BiasInitFn` type alias describes function, with which
    you can initialize `bias` vector for the `Layer`.

    First argument is the size of bias vector, the second is the index in that vector.

    You may want to initialize bias with constant value (`const $ const 0.0`) or
    with random values.
-}
type BiasInitFn = Int -> Int -> Double
{-
    `LayerConfiguration` type alias represents info about layer that is needed for construction.
-}
type LayerConfiguration = (Int, WeightsInitFn, BiasInitFn, ActivationFn)
{-
    `assembleModel` function creates correct sequential model
    with layers of specified sizes and with specified activation functions.

    'Correct sequential model' means that sizes of weights and biases of all layers are compatible
    (constrains that are described in the `weights` and `bias` functions of `Layer` datatype were satisfied).

    As a first argument, the sizing of the input should be supplied, making it essentially an input layer.
-}
assembleModel :: Int -> [LayerConfiguration] -> SequentialModel
assembleModel input_size xs = SequentialModel $ V.fromList $ go input_size xs
 where
    newLayer :: (Int, Int) -> WeightsInitFn -> BiasInitFn -> ActivationFn -> Layer
    newLayer (prev_size, layer_size) w_fn b_fn a_fn = layer
     where
        layer_weights = M.matrix prev_size layer_size (w_fn (prev_size, layer_size))
        layer_bias = V.generate layer_size (b_fn layer_size)
        layer = Layer layer_weights layer_bias a_fn

    go :: Int -> [LayerConfiguration] -> [Layer]
    go _ [] = []
    go prev_size ((size, w_fn, b_fn, a_fn):other_layers) = newLayer (prev_size, size) w_fn b_fn a_fn : go size other_layers

{-
    `BackPropagationStepData` type alias represents data on a step
    of back propagation.
    Data is represented by two matrices -
    first is the result of matrix multiplication and addition and
    the second one is the result of activation function application on the first matrix.
-}
type BackPropagationStepData = (Matrix Double, Matrix Double)
{-
    `_predict'` function processes `(n, i)` matrix through the layers and, saving all the intermediate steps,
    returns the `(n, o)` output,
    where `i = inputSize model` and `o = outputSize model`.
    If the `input` matrix does not have `inputSize model` columns, the error will be thrown.

    Neural network is basically a transformation of a `x`, where `x` is a vector `[x_1, x_2, ..., x_i]`
    (`x` has `i` characteristics). But sometimes you might need to process several `x`s simultaneously
    (for example making use of parallelization). Packing of `n` `x`s in a matrix and processing them
    through the neural network does the job.

    This function usually is not what you want to use - `predict` and `predict'` are the functions for user's usage.
    `_predict'` specializes on memorizing intermediate steps which then are used for the back propagation of error.
    First matrix in the first back propagation step is always empty matrix.
-}
_predict' :: [Layer] -> Matrix Double -> Vector BackPropagationStepData
_predict' model_layers input = V.fromList $ scanl step (M.zero 0 0, input) model_layers
 where
    step :: BackPropagationStepData -> Layer -> BackPropagationStepData
    step (_, prev) (Layer w b act_fn) = (predicted, activated)
     where
        predicted = prev * w + M.matrix (M.nrows prev) (M.ncols w) (\(_, j) -> b V.! (j - 1))
        activated = M.mapPos (const $ call act_fn) predicted

{-
    `predict'` function processes `(n, i)` matrix through the neural network and returns the `(n, o)` output,
    where `i = inputSize model` and `o = outputSize model`.
    If the `input` matrix does not have `inputSize model` columns, the error will be thrown.

    Neural network is basically a transformation of a `x`, where `x` is a vector `[x_1, x_2, ..., x_i]`
    (`x` has `i` characteristics). But sometimes you might need to process several `x`s simultaneously
    (for example making use of parallelization). Packing of `n` `x`s in a matrix and processing them
    through the neural network does the job.

    If you do not want to process several `x`s at the same time, you should prefer using `predict` function.
-}
predict' :: SequentialModel -> Matrix Double -> Matrix Double
predict' model input = snd $ V.last $ _predict' (V.toList $ layers model) input
{-
    `predict` function processes vector with length `i` through the neural network
    and returns the output as a vector of `o` size,
    where `i = inputSize model` and `o = outputSize model`.
    If the `input` matrix does not have `inputSize model` columns, the error will be thrown.

    If you want to process several vectors at the same time, you should prefer using `predict'` function.
-}
predict :: SequentialModel -> Vector Double -> Vector Double
predict model input = M.getRow 1 $ predict' model (M.rowVector input)

{-
    `saveModel` function saves the model data to file for it to be restored later.
-}
saveModel :: SequentialModel -> FilePath -> IO ()
saveModel model filename = do
    file <- openFile filename WriteMode

    let write = hPutStrLn file
    let saveLayer layer = do
            _ <- write $ show $ weights layer
            _ <- write $ show $ bias layer
            _ <- write $ show $ activation_fn layer
            _ <- write ""
            return ()
    
    V.forM_ (layers model) saveLayer
