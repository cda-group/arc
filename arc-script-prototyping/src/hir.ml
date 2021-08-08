open Utils

type hir = (path * item) list

(* and state_param = param *)
(* and event_param = param *)
and state_func = param * param option * block
(* and state_data = path *)

(* and generic = name *)
and name = string
and path = name list
and tv = int  (* Type variable *)
and param = name * ty
and index = int
and 't field = name * 't
and ssa = var * ty * expr
and var = name
and generic = name
and block = ssa list * var
and interface = path * ty list
and subst = (name * ty) list
and item =
  | IVal        of ty * block
  | IEnum       of generic list * path list
  | IExternFunc of generic list * param list * ty
  | IExternType of generic list
  | IFunc       of generic list * param list * ty * block
  | ITask       of generic list * param list * interface * interface * block
  | ITypeAlias  of generic list * ty
  | IVariant    of ty

and ty =
  | TFunc      of ty list * ty
  | TRecord    of ty
  | TRowEmpty
  | TRowExtend of ty field * ty
  | TNominal   of path * ty list
  | TGeneric   of name
  | TArray     of ty
  | TStream    of ty
  | TVar       of name

and binop =
  | BAdd
  | BAnd
  | BBand
  | BBor
  | BBxor
  | BDiv
  | BEq
  | BGeq
  | BGt
  | BLeq
  | BLt
  | BMod
  | BMul
  | BNeq
  | BOr
  | BPow
  | BSub
  | BXor
  | BIn
  | BRExc
  | BRInc
  | BWith

and expr =
  | EAccess   of var * name
  | EAfter    of var * block
  | EEvery    of var * block
  | EArray    of var list
  | EBinOp    of binop * var * var
  | ECall     of var * var list
  | ECast     of var * ty
  | EEmit     of var
  | EEnwrap   of path * ty list * var
  | EIf       of var * block * block
  | EIs       of path * ty list * var
  | ELit      of Ast.lit
  | ELog      of var
  | ELoop     of block
  | EReceive
  | ESelect   of var * var
  | ERecord   of var field list
  | EUnOp     of Ast.unop * var
  | EUnwrap   of path * ty list * var
  | EReturn   of var
  | EBreak    of var
  | EContinue
  (* NB: These expressions are constructed by lowering *)
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

and nominal xs gs = TNominal (xs, gs)

and atom x = TNominal ([x], [])

and parent xs = xs |> rev |> tl |> rev

(* Map types *)
let rec tmap_func f (ps, t, b) =
  let ps = ps |> tmap_params f in
  let t = t |> f in
  let b = b |> tmap_block f in
  (ps, t, b)

and tmap_task f (ps, ts0, ts1, b) =
  let ps = ps |> tmap_params f in
  let ts0 = ts0 |> map f in
  let ts1 = ts1 |> map f in
  let b = b |> tmap_block f in
  (ps, ts0, ts1, b)

and tmap_interface f (xs, ts) =
  let ts = ts |> map f in
  (xs, ts)

and tmap_block f (ss, v) =
  let ss = ss |> map (tmap_ssa f) in
  (ss, v)

and tmap_ssa f (v, t, e) =
  let t = t |> f in
  let e = e |> tmap_expr f in
  (v, t, e)

and tmap_expr f e =
  match e with
  | EIf (v, b0, b1) -> EIf (v, b0 |> tmap_block f, b1 |> tmap_block f)
  | ELoop b -> ELoop (b |> tmap_block f)
  | EEvery (v, b) -> EEvery (v, b |> tmap_block f)
  | EAfter (v, b) -> EAfter (v, b |> tmap_block f)
  | _ -> e

and tmap_params f ps =
  ps |> map (fun (x, t) -> (x, t |> f))

(* Map SSAs *)
let rec smap_item f i =
  match i with
  | IFunc (gs, ps, t, b) -> IFunc (gs, ps, t, b |> smap_block f)
  | ITask (gs, ps, i0, i1, b) -> ITask (gs, ps, i0, i1, b |> smap_block f)
  | _ -> i

and smap_func f (ps, t, b) =
  let b = b |> smap_block f in
  (ps, t, b)

and smap_task f (ps, ts0, ts1, b) =
  let b = b |> smap_block f in
  (ps, ts0, ts1, b)

and smap_block f (ss, v) =
  let ss = ss |> map f in
  (ss, v)

and smap_expr f e =
  match e with
  | EIf (v, b0, b1) -> EIf (v, b0 |> smap_block f, b1 |> smap_block f)
  | ELoop b -> ELoop (b |> smap_block f)
  | EEvery (v, b) -> EEvery (v, b |> smap_block f)
  | EAfter (v, b) -> EAfter (v, b |> smap_block f)
  | _ -> e
