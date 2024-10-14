{-
    `PINN.Model` submodule implements simple sequential neural network model,
    which is going to be used to solve Duffing equation. 
-}
module PINN.Model where

import System.IO

import Data.Vector (Vector)
import qualified Data.Vector as V

import Data.Matrix (Matrix)
import qualified Data.Matrix as M

{-
    `ActivationFn` enum lists activation functions that are supported.

    `getFn` and `getDerivative` function allow to extract useful information from this enum.
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
{-
    `getFn` function returns Haskell representation of the activation function.
-}
getFn :: ActivationFn -> (Double -> Double)
getFn fn = case fn of
            Id -> id
            ReLU -> max 0
            Sin  -> sin
{-
    `getDerivative` function returns Haskell representation of the derivative of activation function.
-}
getDerivative :: ActivationFn -> (Double -> Double)
getDerivative fn = case fn of
                    Id   -> const 1.0
                    ReLU -> fromIntegral . fromEnum . (>= 0)
                    Sin  -> cos

{-
    `WeightsInitFn` type alias describes function, with which
    you can initialize `weights` matrix for the `Layer`.

    You may want to initialize matrix with the same value (`const val`),
    or with random starting values. 
-}
type WeightsInitFn = (Int, Int) -> Double
{-
    `BiasInitFn` type alias describes function, with which
    you can initialize `bias` vector for the `Layer`.

    You may want to initialize vector with the same value (`const val`),
    or with random starting values. 
-}
type BiasInitFn = Int -> Double

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
    `LayerConfiguration` type alias represents info about layer that is needed for construction.
-}
type LayerConfiguration = (Int, WeightsInitFn, BiasInitFn, ActivationFn)
{-
    `assembleModel` function creates correct sequential model
    with layers of specified sizes and with specified activation functions. 

    'Correct sequential model' means that sizes of weights and biases of all layers are compatible
    (constrains that are described in the `weights` and `bias` functions of `Layer` datatype were satisfied).
-}
assembleModel :: [LayerConfiguration] -> SequentialModel
assembleModel xs = SequentialModel $ V.fromList $ go 1 xs
 where
    newLayer :: (Int, Int) -> WeightsInitFn -> BiasInitFn -> ActivationFn -> Layer
    newLayer (prev_size, layer_size) w_fn b_fn a_fn = layer
     where
        layer_weights = M.matrix prev_size layer_size w_fn
        layer_bias = V.generate layer_size b_fn
        layer = Layer layer_weights layer_bias a_fn

    go :: Int -> [LayerConfiguration] -> [Layer]
    go _ [] = []
    go prev_size ((size, w_fn, b_fn, a_fn):other_layers) = newLayer (prev_size, size) w_fn b_fn a_fn : go size other_layers

{-
    `predict` function uses neural network to predict output values
    based on the state of the model and input values.

    `input` vector length must be equal to `inputSize` of model.
    Resulting vector length is equal to `outputSize` of model.
-}
predict :: SequentialModel -> Vector Double -> Vector Double
predict model input = M.getMatrixAsVector $ V.foldl' step (M.rowVector input) (layers model)
 where
    step :: Matrix Double -> Layer -> Matrix Double
    step prev layer = M.mapPos (const $ getFn $ activation_fn layer) (prev * weights layer + M.rowVector (bias layer))

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
