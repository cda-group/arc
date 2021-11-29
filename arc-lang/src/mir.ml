
type mir = ((Hir.path * ty list) * item) list

and name = string
and path = name list
and param = name * ty
and 't field = name * 't
and ssa = var * ty * expr
and var = name
and generic = name
and block = ssa list * var
and interface = path * ty list

and item =
  | IVal         of ty * block
  | IEnum        of path list
  | IExternDef   of ty list * ty
  | IExternType
  | IDef         of param list * ty * block
  | ITask        of param list * interface * interface * block
  | IVariant     of ty

and ty =
  | TFunc      of ty list * ty
  | TRecord    of ty field list
  | TNominal   of path * ty list

and expr =
  | EAccess   of var * name
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
  | EItem     of path * ty list

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

