namespace FNet

[<AutoOpen>]
module Exceptions =

    exception InconsistentModelException of string

    exception BatchTooBigForDataSourceException

