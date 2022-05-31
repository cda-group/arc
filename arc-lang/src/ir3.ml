open Error

type ir3 = ((path * tys) * item) list
and names = name list
and name = string
and paths = path list
and path = names
and params = param list
and param = name * ty
and index = int
and 't fields = 't field list
and 't field = name * 't
and 't variants = 't variant list
and 't variant = name * 't
and block = stmts * var
and vars = var list
and var = name

and stmts = stmt list
and stmt = SVal of name * ty * expr

and decorator = Ast.decorator

and async = bool

and items = item list
and item =
  | IDef         of loc * decorator * params * ty * block
  | IExternDef   of loc * decorator * async * tys * ty
  | IExternType  of loc * decorator
  | IType        of loc * decorator * ty
  | IVal         of loc * decorator * ty * block

and tys = ty list
and ty =
  | TFunc      of tys * ty
  | TRecord    of ty fields
  | TEnum      of ty variants
  | TNominal   of path * tys

and lit = Ast.lit

and exprs = expr list
and expr =
  | EAccess   of loc * var * name
  | EBreak    of loc * var
  | ECallExpr of loc * var * vars
  | ECallItem of loc * path * tys * vars
  | ECast     of loc * var * ty
  | EContinue of loc
  | EEnwrap   of loc * name * var
  | EUnwrap   of loc * name * var
  | ECheck    of loc * name * var
  | EItem     of loc * path * tys
  | EIf       of loc * var * block * block
  | ELit      of loc * lit
  | ELoop     of loc * block
  | ERecord   of loc * var fields
  | EReturn   of loc * var
  | EUpdate   of loc * var * name * var
  | ESpawn    of loc * path * tys * vars

let item_loc i =
  match i with
  | IExternDef   (i, _, _, _, _) -> i
  | IDef         (i, _, _, _, _) -> i
  | IVal         (i, _, _, _) -> i
  | IExternType  (i, _) -> i
  | IType        (i, _, _) -> i

(* Returns the parent path of a path *)
let rec parent xs = xs |> List.rev |> List.tl |> List.rev

and atom x = TNominal (["std"; x], [])

and is_unit t =
  t = (atom "unit")
