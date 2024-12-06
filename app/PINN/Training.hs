{-
    `PINN.Training` submodule provides training functions for the model -
    it supplies some convenience functions that help creating the dataset
    and implements batching which makes the training more efficient.
-}
module PINN.Training where

import Control.Monad
import Control.Monad.ST
import Data.STRef

import System.Random

import Data.Vector (Vector, (!))
import qualified Data.Vector as V

import Data.Vector.Mutable (MVector, PrimMonad, PrimState)
import qualified Data.Vector.Mutable as MV

import Data.Matrix (Matrix)
import qualified Data.Matrix as M

import PINN.Differentiation (DifferentiableFn(..), LossFn)
import PINN.Model (Layer(..), SequentialModel(..), _backpropagationPredict)
import PINN.Optimisers (Optimiser(..))

{-
    `Dataset` type alias represents the dataset that is needed for the training -
    it is a vector of input values and true output values.
-}
type Dataset = Vector (Vector Double, Vector Double)
{-
    `Batch` type alias represents one batch (several vectors of input and true output values grouped together in a matrices).
-}
type Batch = (Matrix Double, Matrix Double)
{-
    `shuffleDataset` returns dataset where all items have been shuffled.
    
    This function accepts more generic type `Vector a`, so it can shuffle any vector, not necessarily a `Dataset`.
-}
shuffleDataset :: (Vector a, StdGen) -> (Vector a, StdGen)
shuffleDataset (dataset, seed0) = runST $ do
    mutable_vector <- V.thaw dataset
    new_seed <- shuffle mutable_vector seed0
    vector <- V.unsafeFreeze mutable_vector
    return (vector, new_seed)
 where
    shuffle :: (PrimMonad s) => MVector (PrimState s) a -> StdGen -> s StdGen
    shuffle vector = go (MV.length vector)
     where
        go size seed = if size <= 1 then return seed
                       else do
                           let (swap_i, new_seed) = randomR (0, size - 1) seed
                           _ <- MV.swap vector swap_i (size - 1)
                           go (size - 1) new_seed
{-
    `splitToBatches` function splits dataset to a list of smaller datasets (batches).
-}
splitToBatches :: Int -> Dataset -> Vector Batch
splitToBatches size dataset = V.fromList $ go dataset
 where
    groupInBatch :: Vector (Vector Double, Vector Double) -> Batch
    groupInBatch vector = (tl, tr)
     where
        rows = V.length vector
        input_cols = V.length $ fst $ V.head vector

        group (i', j') = (if j < input_cols then fst else snd) (vector ! i) ! (j `mod` input_cols)
         where
            i = i' - 1
            j = j' - 1

        (tl, tr, _, _) = M.splitBlocks rows input_cols (M.matrix rows (input_cols + V.length (snd $ V.head vector)) group)

    go vector = if V.length vector <= size then [groupInBatch vector]
                else
                    groupInBatch current : go remainder
     where
        (current, remainder) = V.splitAt size vector

{-
    `TrainingHyperparameters` record datatype represents parameters that do not depend
    on the training itself, and are basically parameters of the training.
-}
data TrainingHyperparameters = TrainingHyperparameters {
    {-
        `epochs` is a number that defines how many times the same dataset will be processed.
    -}
    epochs :: Int,
    {-
        `learningRate` is a function that takes the current iteration of training 
        and returns a coefficient with which gradients are applied.

        Taking the current iteration of training allows to make a dynamic learning rate
        (for example, decay).
    -}
    learningRate :: Int -> Double,

    {-
        `lossFn` is a differentiable function that represents the offset between predicted value and a true one.
    -}
    lossFn :: LossFn,

    {-
        `batchSize` controls the size of each batch to which the dataset is splitted.
    -}
    batchSize :: Int
}

{-
    `train` function is the most important function for neural network - it is a function that
    trains the network on the supplied dataset using optimiser, which helps to correct the weights and biases of model.
-}
train
    :: (Optimiser optimiser params) =>
       SequentialModel
    -> optimiser
    -> (Dataset, StdGen)
    -> TrainingHyperparameters
    -> (SequentialModel, Vector Double, StdGen)
train model optimiser (dataset, seed0) (TrainingHyperparameters e lr loss size) = runST $ do
    model_layers <- V.thaw $ layers model
    layer_parameters <- V.thaw $ initializeParams optimiser model

    losses <- MV.new (e * (V.length dataset `div` size))

    seed <- newSTRef seed0
    iteration <- newSTRef 0

    forM_ [1..e] $ \_ -> do
        current_seed <- readSTRef seed
        let (shuffled_dataset, new_seed) = shuffleDataset (dataset, current_seed)
        writeSTRef seed new_seed

        let batches = splitToBatches size shuffled_dataset
        forM_ batches $ \(batch_input, batch_output) -> do
            i <- readSTRef iteration

            freezed_model <- V.freeze model_layers
            let back_propagation = _backpropagationPredict (V.toList freezed_model) batch_input
            let predicted_output = snd $ V.last back_propagation

            let batches_rows = [(M.getRow row batch_output, M.getRow row predicted_output) | row <- [1..(M.nrows batch_output)]]
            MV.write losses i (sum $ map (V.sum . call loss) batches_rows)
            let loss_derivative = M.fromLists $ map (V.toList . derivative loss) batches_rows

            let lr_k = lr i

            modifySTRef' iteration (+1)

            layer_error <- newSTRef loss_derivative
            forM_ (reverse [0..(MV.length model_layers - 1)]) $ \layer_index -> do
                current_layer <- MV.read model_layers layer_index

                dEdH <- readSTRef layer_error
                let dEdT = M.elementwise (*) dEdH (M.mapPos (const $ derivative $ activationFn current_layer) (fst $ back_propagation ! (layer_index + 1)))
                let dEdW = M.transpose (snd $ back_propagation ! layer_index) * dEdT
                let dEdB = M.getRow 1 dEdT
                writeSTRef layer_error (dEdT * M.transpose (weights current_layer))

                current_params <- MV.read layer_parameters layer_index
                let (new_layer, new_params) = updateStep optimiser current_params lr_k (dEdW, dEdB) current_layer

                MV.write model_layers layer_index new_layer
                MV.write layer_parameters layer_index new_params
            

    last_seed <- readSTRef seed
    total_losses <- V.unsafeFreeze losses
    updated_layers <- V.unsafeFreeze model_layers

    return (SequentialModel updated_layers, total_losses, last_seed)
    