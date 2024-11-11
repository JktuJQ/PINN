module Main where

import Data.Bifunctor
import System.Random

import Graphics.EasyPlot

import Data.Vector ((!))
import qualified Data.Vector as V

import qualified Data.Matrix as M

import Task (differentialEquations)

import CauchyProblem (CauchyData(..))
import CauchyProblem.Times (timeline, TimeSettings(..), fromTimeSettings)
import CauchyProblem.NumericalMethods.Methods (methodRungeKutta)

import PINN.Model
import PINN.DifferentiableFns
import PINN.Optimisers
import PINN.Training

main :: IO ()
main = do
    -- XOR approximation
    putStrLn "XOR approximation"
    let dataset = V.fromList [(V.fromList [0.0, 0.0], V.singleton 0.0), (V.fromList [1.0, 0.0], V.singleton 1.0), (V.fromList [0.0, 1.0], V.singleton 1.0), (V.fromList [1.0, 1.0], V.singleton 0.0)]
    let model = assembleModel 2 [
            (2, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Sin)
                                ]
    let sgd = SGD 0.5 False
    let (trained_model, losses, _) = train model sgd
                                (dataset, mkStdGen 1)
                                (TrainingHyperparameters 100 (const 0.01) SSR 4)
    _ <- plot (PNG "tests/losses_xor.png") (Data2D [Title "loss", Style Lines] [] [(fromIntegral i, losses ! i) | i <- [0..(V.length losses - 1)]])

    -- x^2 approximation
    putStrLn "x2 approximation"
    let timegrid = fromTimeSettings (TimeSettings (-1.0) 1.0 (const 0.1))
    let cauchy_data = CauchyData (V.fromList [1.0]) (V.fromList [\(t, _) -> 2.0 * t])
    let result = methodRungeKutta timegrid cauchy_data
    let dataset = V.map (\t -> (V.singleton t, V.singleton (t * t))) (V.fromList $ timeline timegrid)

    let model = assembleModel 1 [
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Id)
                                ]
    let sgd = SGD 0.5 False
    let (trained_model, losses, _) = train model sgd
                                (dataset, mkStdGen 1)
                                (TrainingHyperparameters 1000 (const 0.01) SSR 1)
    _ <- plot (PNG "tests/predicted_x2.png") (Data2D [Title "predicted", Style Lines] [] [(i, V.head $ predict trained_model (V.singleton i)) | i <- [-1.0, -0.9..1.0]])
    _ <- plot (PNG "tests/losses_x2.png") (Data2D [Title "loss", Style Lines] [] [(fromIntegral i, losses ! i) | i <- [0..(V.length losses - 1)]])

    -- sin(2.0 * cos(x) + 3.0) + 2.5 approximation
    putStrLn "sin(2.0 * cos(x) + 3.0) + 2.5 approximation"
    let timegrid = fromTimeSettings (TimeSettings (-(2.0 * pi)) (2.0 * pi) (const 0.1))
    let cauchy_data = CauchyData (V.fromList [1.54]) (V.fromList [\(t, _) -> (-(2.0 * sin t * cos (2.0 * cos t + 3.0)))])
    let result = methodRungeKutta timegrid cauchy_data
    let dataset = V.map (\t -> (V.singleton t, V.singleton (sin(2.0 * cos t + 3.0) + 2.5))) (V.fromList $ timeline timegrid)

    let model = assembleModel 1 [
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Id)
                                ]
    let sgd = SGD 0.0 False
    let (trained_model, losses, _) = train model sgd
                                (dataset, mkStdGen 1)
                                (TrainingHyperparameters 1000 (const 0.001) SSR 1)
    _ <- plot (PNG "tests/predicted_sincos_trig_fn.png") (Data2D [Title "predicted", Style Lines] [] [(i, V.head $ predict trained_model (V.singleton i)) | i <- [-(2.0 * pi), (-(2.0 * pi) + 0.1)..(2.0 * pi)]])
    _ <- plot (PNG "tests/losses_sincos_trig_fn.png") (Data2D [Title "loss", Style Lines] [] [(fromIntegral i, losses ! i) | i <- [0..(V.length losses - 1)]])

    return ()
