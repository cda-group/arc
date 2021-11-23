open Ast

module PathMap = Map.Make(
  struct
    type t = path
    let compare = Stdlib.compare
  end
)

type decl =
  | DItem of kind
  | DUse of path (* An alias *)
and kind =
  | DEnum
  | DExternDef
  | DExternType
  | DDef
  | DClass
  | DTask
  | DTypeAlias
  | DVariant
  | DGlobal
  | DMod
and name = string
and path = name list
and table = decl PathMap.t
