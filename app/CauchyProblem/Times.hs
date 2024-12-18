{-
    `CauchyProblem.Times` submodule supplies definitions for time representation.
-}
module CauchyProblem.Times where

{-
    `Time` type alias corresponds to a numerical value of time parameter.
-}
type Time = Double
{-
    `Timeline` type alias represents a line of time values.

    You can think of a timeline as a slice of `Timegrid`.
-}
type Timeline = [Time]

{-
    `TimeStepFn` type alias represents function that specifies the time interval that can change.
-}
type TimeStepFn = Int -> Time

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
        `stepFn` function specifies what time interval will be separating values in a `Timegrid`.
        
        `stepFn i` returns time interval that should pass between i-th and i+1-th time values,
        thus, it should be strictly positive.

        This function can be used to create `Timegrid`s with variable step, but it is still
        very easy to implement `stepFn` for function with constant step - `stepFn = const tau`.
    -}
    stepFn :: TimeStepFn
}

{-
    `Timegrid` record datatype represents a timegrid on which numerical methods will operate.

    It holds its initial time settings, because those are needed for the entirety of timegrid usage.
-}
data Timegrid = Timegrid {
    {-
        `tauFn` function specifies what time interval will be separating values in a `Timegrid`.

        `Timegrid` obtains this function from the `TimeSettings` `stepFn`.
    -}
    tauFn :: TimeStepFn,

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
fromTimeSettings settings = Timegrid step (takeWhile (<= end) (create start 0))
 where
    start = t0 settings
    end = t1 settings
    step = stepFn settings

    create :: Time -> Int -> [Time]
    create prev i = prev : create (prev + step i) (i + 1)
