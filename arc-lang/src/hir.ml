open Utils

type hir = (path * item) list

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
  | IEnum        of generic list * path list
  | IExternDef   of generic list * ty list * ty
  | IExternType  of generic list
  | IDef         of generic list * param list * ty * block
  | IClassDecl   of path * generic list * param list * ty
  | IInstanceDef of path * generic list * param list * ty * block
  | IClass       of generic list
  | IInstance    of generic list * path * ty list
  | ITask        of generic list * param list * interface * interface * block
  | ITypeAlias   of generic list * ty
  | IVariant     of ty

and ty =
  | TFunc      of ty list * ty
  | TRecord    of ty
  | TRowEmpty
  | TRowExtend of ty field * ty
  | TNominal   of path * ty list
  | TGeneric   of name
  | TVar       of name

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

let nominal xs gs = TNominal (xs, gs)

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
  | _ -> e

and tmap_params f ps =
  ps |> map (fun (x, t) -> (x, t |> f))

(* Map SSAs *)
let rec smap_item f i =
  match i with
  | IDef (gs, ps, t, b) -> IDef (gs, ps, t, b |> smap_block f)
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
  | _ -> e

(* Conversions *)

let index_to_field i = Printf.sprintf "_%d" i

let indexes_to_fields is =
  is |> List.fold_left (fun (l, c) v -> ((index_to_field c, v)::l, c+1)) ([], 0)
     |> fst
     |> List.rev

let arms_to_clauses arms v =
  arms |> List.map (fun (p, e) -> ([(v, p)], [], e))

(* t is the tail, which could either be a Hir.TVar or Hir.TRowEmpty *)
let fields_to_rows t fs =
  fs |> List.fold_left (fun t f -> TRowExtend (f, t)) t

let indexes_to_rows t is =
  is |> indexes_to_fields |> fields_to_rows t
