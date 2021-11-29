open Hir

type mir = ((path * ty list) * item) list
and item =
  | IVal         of ty * block
  | IEnum        of path list * ty list
  | IExternDef   of ty list * ty
  | IExternType
  | IDef         of param list * ty * block
  | IClassDecl   of path * param list * ty
  | IInstanceDef of path * param list * ty * block
  | IClass
  | IInstance    of path * ty list
  | ITask        of param list * interface * interface * block

and ty =
  | TFunc      of ty list * ty
  | TRecord    of ty field list
  | TNominal   of path * ty list

and expr =
  | EAccess   of var * name
  | EAfter    of var * block
  | EEvery    of var * block
  | EEq       of var * var
  | ECall     of var * var list
  | ECast     of var * ty
  | EEmit     of var
  | EEnwrap   of path * ty list * var
  | EIf       of var * block * block
  | EIs       of path * ty list * var
  | ELit      of Ast.lit
  | ELoop     of block
  | EReceive
  | ERecord   of var field list
  | EUnwrap   of path * ty list * var
  | EReturn   of var
  | EBreak    of var
  | EContinue
  (* NB: These expressions are constructed by lowering *)
  | EItem     of path * ty list
