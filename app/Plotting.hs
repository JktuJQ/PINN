{-
    `Plotting` module provides plotting API for this project.

    Goal of this project is to solve Duffing equation using PINN and numerical methods,
    and it is very important for the research to be able to visualize approximated functions.
    This exact task may require all sorts of plots:
    logarithmic scale plot to test numerical methods order,
    static plots (both 2d and 3d) of solution that express impact of various parameters values
    and visualize the oscillator itself by its position or phase portrait,
    dynamic plotting of Poincaré section and attractor of Duffing equation
    and many more.

    This module uses `easyplot` Haskell dependency and `gnuplot` as an external
    dependency. Although `easyplot` provides simple API that is enough for most of uses,
    it does not fully meet all requirements of this project. That said, this module
    tries to provide an abstraction layer over `easyplot` and `gnuplot`.
-}
module Plotting where
