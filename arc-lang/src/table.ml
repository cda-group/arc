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
  | DEnum of arity
  | DExternDef of arity
  | DExternType of arity
  | DDef of arity
  | DClass of arity
  | DTask of arity
  | DTypeAlias of arity * generics * ty
  | DVariant of arity
  | DGlobal
  | DMod
and arity = int
and name = string
and path = name list
and table = decl PathMap.t
