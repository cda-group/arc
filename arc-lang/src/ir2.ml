open Error

type ir2 = (path * item) list
and names = name list
and name = string
and paths = path list
and path = names
and params = param list
and param = name * ty
and index = int
and 't fields = 't field list
and 't field = name * 't
and 't record = 't fields * 't option
and variants = variant list
and variant = loc * name * tys
and block = stmts * var
and vars = var list
and var = name
and generics = generic list
and generic = name

and stmts = stmt list
and stmt =
  | SVal of name * ty * expr

and decorator = Ast.decorator

and async = bool

and items = item list
and item =
  | IClass       of loc * decorator * generics * paths
  | IClassDef    of loc * decorator * path * generics * params * ty
  | IDef         of loc * decorator * generics * params * ty * block
  | IExternDef   of loc * decorator * async * generics * tys * ty
  | IExternType  of loc * decorator * generics
  | IInstance    of loc * decorator * generics * path * tys * paths
  | IInstanceDef of loc * decorator * path * generics * params * ty * block
  | IType        of loc * decorator * generics * ty
  | IVal         of loc * decorator * ty * block

and tys = ty list
and ty =
  | TFunc      of tys * ty
  | TRecord    of ty
  | TEnum      of ty
  | TRowEmpty
  | TRowExtend of ty field * ty
  | TNominal   of path * tys
  | TGeneric   of name

and lit = Ast.lit

and exprs = expr list
and expr =
  | EAccess   of loc * var * name
  | ESubset   of loc * var * ty
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
  | ERecord   of loc * var record
  | EReturn   of loc * var
  | EUpdate   of loc * var * name * var
  | ESpawn    of loc * path * tys * vars

let item_loc i =
  match i with
  | IExternDef   (i, _, _, _, _, _) -> i
  | IDef         (i, _, _, _, _, _) -> i
  | IVal         (i, _, _, _) -> i
  | IExternType  (i, _, _) -> i
  | IClass       (i, _, _, _) -> i
  | IClassDef    (i, _, _, _, _, _) -> i
  | IInstanceDef (i, _, _, _, _, _, _) -> i
  | IInstance    (i, _, _, _, _, _) -> i
  | IType        (i, _, _, _) -> i

let atom x = TNominal (["std"; x], [])
