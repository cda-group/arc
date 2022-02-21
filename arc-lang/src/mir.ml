
type mir = ((path * tys) * item) list

and name = string
and paths = path list
and path = name list
and params = param list
and param = name * ty
and 't fields = 't field list
and 't field = name * 't
and ssas = ssa list
and ssa = var * ty * expr
and vars = var list
and var = name
and block = ssas * var
and interface = path * tys
and decorator = Ast.decorator

and item =
  | IVal         of decorator * ty * block
  | IEnum        of decorator * paths
  | IExternDef   of decorator * tys * ty
  | IExternType  of decorator
  | IDef         of decorator * params * ty * block
  | ITask        of decorator * params * params * block
  | IVariant     of ty

and tys = ty list
and ty =
  | TFunc      of tys * ty
  | TRecord    of ty fields
  | TNominal   of path * tys

and expr =
  | EAccess   of var * name
  | ECall     of var * vars
  | ECast     of var * ty
  | EEmit     of var * var
  | EEnwrap   of path * tys * var
  | EIf       of var * block * block
  | EIs       of path * tys * var
  | ELit      of Ast.lit
  | ELoop     of block
  | EReceive  of var
  | ERecord   of var fields
  | EUnwrap   of path * tys * var
  | EReturn   of var
  | EBreak    of var
  | EContinue
  | EItem     of path * tys

let is_int t =
  match t with
  | TNominal (["i16" | "i32" | "i64" | "i128"], []) -> true
  | _ -> false

let is_float t =
  match t with
  | TNominal (["f32" | "f64"], []) -> true
  | _ -> false

let is_bool t =
  match t with
  | TNominal (["bool"], []) -> true
  | _ -> false

let is_unit t =
  match t with
  | TNominal (["unit"], []) -> true
  | _ -> false

and nominal xs = TNominal (xs, [])
and atom x = TNominal ([x], [])

and ts_of_vs vs = vs |> List.map (fun (_, t) -> t)
and fts_of_fvs vs = vs |> List.map (fun (x, (_, t)) -> (x, t))

