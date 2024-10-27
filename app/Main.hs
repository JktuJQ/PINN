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
    let timegrid = fromTimeSettings (TimeSettings 0.0 50.0 (const 0.1))
    let cauchy_data = CauchyData (V.fromList [1.0, 0.0]) (differentialEquations (-1.0, 1.0, 0.2, 0.3, 1.2))
    let result = methodRungeKutta timegrid cauchy_data
    let dataset = zip (timeline timegrid) (V.toList $ V.map (! 1) result)
    _ <- plot (PNG "executable/task.png") (Data2D [Title "xt", Style Lines] [] dataset)

    let model = assembleModel 2 [
            (8, const $ const 1.0, const $ const 0.0, Sin),
            (8, const $ const 1.0, const $ const 0.0, ReLU),
            (8, const $ const 1.0, const $ const 0.0, Sin),
            (1, const $ const 1.0, const $ const 0.0, Id)
                                ]
    
    {-
    let sgd = SGD 0.0 False
    let (trained_model, losses, _) = train model sgd
                                (V.fromList [(V.fromList [0.0, 0.0], V.singleton 0.0), (V.fromList [1.0, 0.0], V.singleton 1.0), (V.fromList [0.0, 1.0], V.singleton 1.0), (V.fromList [1.0, 1.0], V.singleton 1.0)], mkStdGen 1)
                                (TrainingHyperparameters 10 (const 0.01) SSR 1)
    _ <- plot (PNG "executable/losses.png") (Data2D [Title "loss", Style Lines] [] [(fromIntegral i, losses ! i) | i <- [0..999]])
    print $ predict trained_model (V.fromList [0.0, 0.0])
    print $ predict trained_model (V.fromList [1.0, 0.0])
    print $ predict trained_model (V.fromList [0.0, 1.0])
    print $ predict trained_model (V.fromList [1.0, 1.0])
    
    V.forM_ (layers trained_model) $ \layer -> do
        print $ weights layer
        print $ M.nrows $ weights layer
        print $ M.ncols $ weights layer

        print $ bias layer
    -}
    

    let sgd = SGD 0.0 False
    let (trained_model, losses, _) = train model sgd
                                (V.map (bimap V.singleton V.singleton) (V.fromList dataset), mkStdGen 1)
                                (TrainingHyperparameters 50 (const 0.01) MSE 128)
    let predicts = [(t, V.head $ predict trained_model (V.singleton t)) | t <- timeline timegrid]
    _ <- plot (PNG "executable/predicted.png") (Data2D [Title "xt", Style Lines] [] predicts)

    V.forM_ (V.fromList dataset) $ \(t, x) -> do
        let predicted = V.head $ predict trained_model (V.singleton t)
        print predicted
        print x
        print (x - predicted)

    print ""
    _ <- plot (PNG "executable/losses.png") (Data2D [Title "loss", Style Lines] [] [(fromIntegral i, losses ! i) | i <- [0..(V.length losses - 1)]])
    
    return ()
