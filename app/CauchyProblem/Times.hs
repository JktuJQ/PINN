{-
    `CauchyProblem.Times` submodule supplies definitions for time representation.
-}
module CauchyProblem.Times where

import Data.Vector (Vector, fromList)

{-
    `Time` type alias corresponds to a numerical value of time parameter.
-}
type Time = Double
{-
    `Timeline` type alias represents a line of time values.

    You can think of a timeline as a slice of `Timegrid`.
-}
type Timeline = Vector Time

{-
    `TimeSettings` record datatype describes settings with which a `Timegrid` will be created.
-}
data TimeSettings = TimeSettings {
    {-
        Initial time value (start point).

        It is guaranteed that created from `TimeSettings` `Timegrid`'s
        first value will be equal to `t0`.
    -}
    t0 :: Time,
    {-
        End point of time.

        `Timegrid` that is created from `TimeSettings` may or may not end with `t1`,
        but it is guaranteed that no value will exceed `t1`.
    -}
    t1 :: Time,

    {-
        `step_fn` function specifies what time interval will be separating values in a `Timegrid`.
        
        `step_fn i` returns time interval that should pass between i-th and i+1-th time values,
        thus, it should be strictly positive.

        This function can be used to create `Timegrid`s with variable step, but it is still
        very easy to implement `step_fn` for function with constant step - `step_fn = const tau`.
    -}
    step_fn :: Int -> Time
}

{-
    `Timegrid` record datatype represents a timegrid on which numerical methods will operate.

    It holds its initial time settings, because those are needed for the entirety of timegrid usage.
-}
data Timegrid = Timegrid {
    {-
        `TimeSettings` with which this `Timegrid` was created.

        It is sometimes necessary to still have initial values
        after iterating on a corresponding timeline.
    -}
    time_settings :: TimeSettings,

    {-
        `Timeline` which represents this `Timegrid`.

        `Timeline` is only a slice by definition - timeline to which
        this field refers may be already empty after iteration.
    -}
    timeline :: Timeline
}

{-
    Constructs `Timegrid` from given `TimeSettings`.
-}
fromTimeSettings :: TimeSettings -> Timegrid
fromTimeSettings settings = Timegrid settings (fromList $ takeWhile (\t -> t <= end) (create start 0))
 where
    start = t0 settings
    end = t1 settings
    step = step_fn settings

    create prev i = prev : create (prev + step i) (i + 1)
