module Main where

import PINN.Testing

main :: IO ()
main = do
    let (_, xor) = testXOR
    let (_, x_squared) = testXSquared
    let (_, trigonometry) = testTrigonometry

    mapM_ (\(f, p) -> f p) (zip (map plotTest ["xor", "x2", "trigonometry"]) [xor, x_squared, trigonometry])
