namespace FNet

open System

[<AutoOpen>]
module NetworkFunctions =

    let Logistic (x: double) = (double) 1.0 / (1.0 + Math.Exp -x)

    let LogisticDerivative_f_output (y: double) = y * (1.0 - y)

