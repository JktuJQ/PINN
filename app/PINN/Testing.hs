{-
    `PINN.Testing` submodule provides several tests for neural networks.
    Most of the time those tests try to approximate some functions with different neural network configurations.
-}
module PINN.Testing where

import System.Random

import Graphics.EasyPlot

import Data.Vector ((!))
import qualified Data.Vector as V

import PINN.Model
import PINN.Differentiation
import PINN.Optimisers
import PINN.Training

{-
    `TestPlots` record datatype holds several plots that test may or may not provide.

    It is needed to pack those plots and then supply them to the `plotTest` function.
-}
data (Plot a) => TestPlots a = TestPlots {
    {-
        Original function (analytic plot).
    -}
    original :: Maybe a,
    {-
        Predicted function (plot of predicted values).
    -}
    prediction :: Maybe a,
    {-
        Plot of neural networks losses.
    -}
    losses :: Maybe a
}
{-
    `plotTest` function plots all data from `TestPlots` datatype.

    If present, plots will be named as follows: 'tests/{plot_type (original, prediction, losses)}_{plot_name}.png'.
-}
plotTest :: (Plot a) => String -> TestPlots a -> IO ()
plotTest name (TestPlots orig pr loss) = do
    let plot_names = ["tests/" ++ plot_name ++ "_" ++ name ++ ".png" | plot_name <- ["original", "predicted", "losses"]]
    mapM_ (\(str, p) -> maybe (return False) (plot (PNG str)) p) (zip plot_names [orig, pr, loss])

{-
    `testXOR` function tries to approximate `XOR` function (0 0 -> 0, 0 1 -> 1, 1 0 -> 1, 1 1 -> 0).
-}
testXOR :: (SequentialModel, TestPlots (Graph2D Double Double))
testXOR = (trained_model, TestPlots Nothing Nothing (Just losses_plot))
 where
    dataset = V.fromList [
        (V.fromList [0.0, 0.0], V.singleton 0.0),
        (V.fromList [1.0, 0.0], V.singleton 1.0),
        (V.fromList [0.0, 1.0], V.singleton 1.0),
        (V.fromList [1.0, 1.0], V.singleton 0.0)
                         ]

    model = assembleModel 2 [
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Sin)
                            ]

    optimiser = SGD 0.5 False
    hyperparams = TrainingHyperparameters 100 (const 0.01) SSR 4
    (trained_model, loss, _) = train model optimiser (dataset, mkStdGen 1) hyperparams

    losses_plot = Data2D [Title "loss", Style Lines] [] [(fromIntegral i, loss ! i) | i <- [0..(V.length loss - 1)]]

{-
    `testXSquared` function tries to approximate `x^2` function.
-}
testXSquared :: (SequentialModel, TestPlots (Graph2D Double Double))
testXSquared = (trained_model, TestPlots (Just original_plot) (Just predicted_plot) (Just losses_plot))
 where
    range = [(-1.0),(-0.9)..1.0]

    dataset = V.map (\t -> (V.singleton t, V.singleton (t * t))) (V.fromList range)

    model = assembleModel 1 [
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Id)
                            ]

    optimiser = SGD 0.5 True
    hyperparams = TrainingHyperparameters 1000 (const 0.01) SSR 1
    (trained_model, loss, _) = train model optimiser (dataset, mkStdGen 1) hyperparams

    original_plot = Function2D [Title "predicted", Style Lines] [For range] (\x -> x * x)
    predicted_plot = Data2D [Title "predicted", Style Lines] [Range (-1.0) 1.0] [(i, V.head $ predict trained_model (V.singleton i)) | i <- range]
    losses_plot = Data2D [Title "loss", Style Lines] [] [(fromIntegral i, loss ! i) | i <- [0..(V.length loss - 1)]]

{-
    `testTrigonometry` function tries to approximate `sin(2.0 * cos(x) + 3.0) + 2.5` function.
-}
testTrigonometry :: (SequentialModel, TestPlots (Graph2D Double Double))
testTrigonometry = (trained_model, TestPlots (Just original_plot) (Just predicted_plot) (Just losses_plot))
 where
    range = [(-(2.0 * pi)),((-(2.0 * pi)) + 0.1)..(2.0 * pi)]

    dataset = V.map (\t -> (V.singleton t, V.singleton (sin (2.0 * cos t + 3.0) + 2.5))) (V.fromList range)

    model = assembleModel 1 [
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Id)
                            ]

    optimiser = Adam 0.9 0.9
    hyperparams = TrainingHyperparameters 1000 (const 0.001) SSR 1
    (trained_model, loss, _) = train model optimiser (dataset, mkStdGen 1) hyperparams

    original_plot = Function2D [Title "predicted", Style Lines] [Range (-(2.0 * pi)) (2.0 * pi)] (\x -> sin (2.0 * cos x + 3.0) + 2.5)
    predicted_plot = Data2D [Title "predicted", Style Lines] [Range (-(2.0 * pi)) (2.0 * pi)] [(i, V.head $ predict trained_model (V.singleton i)) | i <- range]
    losses_plot = Data2D [Title "loss", Style Lines] [] [(fromIntegral i, loss ! i) | i <- [0..(V.length loss - 1)]]
