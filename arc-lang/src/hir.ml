open Utils

type hir = (path * item) list

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
and generics = generic list
and generic = name
and block = ssas * var
and interface = path * tys
and decorator = Ast.decorator
and item =
  | IVal         of decorator * ty * block
  | IEnum        of decorator * generics * paths
  | IExternDef   of decorator * generics * tys * ty
  | IExternType  of decorator * generics
  | IDef         of decorator * generics * params * ty * block
  | IClassDecl   of decorator * path * generics * params * ty
  | IInstanceDef of decorator * path * generics * params * ty * block
  | IClass       of decorator * generics
  | IInstance    of decorator * generics * path * tys
  | ITask        of decorator * generics * params * params * block
  | ITypeAlias   of decorator * generics * ty
  | IVariant     of ty

and tys = ty list
and ty =
  | TFunc      of tys * ty
  | TRecord    of ty
  | TRowEmpty
  | TRowExtend of ty field * ty
  | TNominal   of path * tys
  | TGeneric   of name
  | TVar       of name

and expr =
  | EAccess   of var * name
  | EUpdate   of var * name * var
  | ECall     of var * vars
  | ECast     of var * ty
  | EEnwrap   of path * tys * var
  | EIf       of var * block * block
  | EIs       of path * tys * var
  | ELit      of Ast.lit
  | ELoop     of block
  | EEmit     of var * var
  | EReceive  of var
  | EOn       of receivers
  | ERecord   of var fields
  | EUnwrap   of path * tys * var
  | EReturn   of var
  | EBreak    of var
  | EContinue
  | EItem     of path * tys

and receivers = receiver list
and receiver = var * var * block

let nominal xs gs = TNominal (xs, gs)

and atom x = TNominal ([x], [])

and parent xs = xs |> rev |> tl |> rev

(* Map types *)
let rec tmap_def f (ps, t, b) =
  let ps = ps |> tmap_params f in
  let t = t |> f in
  let b = b |> tmap_block f in
  (ps, t, b)

and tmap_task f (ps0, ps1, b) =
  let ps0 = ps0 |> tmap_params f in
  let ps1 = ps1 |> tmap_params f in
  let b = b |> tmap_block f in
  (ps0, ps1, b)

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
  | EEnwrap (xs, ts, v) -> EEnwrap (xs, ts |> map f, v)
  | EUnwrap (xs, ts, v) -> EUnwrap (xs, ts |> map f, v)
  | EIs (xs, ts, v) -> EIs (xs, ts |> map f, v)
  | _ -> e

and tmap_params f ps =
  ps |> map (fun (x, t) -> (x, t |> f))

(* Map SSAs *)
let rec smap_item f i =
  match i with
  | IDef (a, gs, ps, t, b) -> IDef (a, gs, ps, t, b |> smap_block f)
  | ITask (a, gs, ps0, ps1, b) -> ITask (a, gs, ps0, ps1, b |> smap_block f)
  | _ -> i

and smap_def f (ps, t, b) =
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

(* Converts a list [v0; v1; ..; vn] into [("_0", v0); ("_1", v1); ...; ("_n", vn)] *)
let indexes_to_rows t is =
  is |> indexes_to_fields |> fields_to_rows t

(* Calculates the free variables of a block `b` parameterized by `ps` *)
and free_vars ps b =

  (* Variables which are not free *)
  let vs = ps |> map (fun (v, _) -> v) in

  let is_free v scopes = not (scopes |> exists (mem v)) in

  let fv_var v (scopes, acc) =
    if is_free v scopes then
      (scopes, v::acc)
    else
      (scopes, acc)
  in

  let fv_vars vs ctx = vs |> foldl (fun ctx v -> fv_var v ctx) ctx in

  let push_scope (scopes, acc) = ([]::scopes, acc) in
  let pop_scope (scopes, acc) = (tl scopes, acc) in

  let def_var v (scopes, acc) =
    match scopes with
    | h::t -> ((v::h)::t, acc)
    | _ -> unreachable ()
  in

  let rec fv_block (ss, v) ctx =
    let ctx = ctx |> push_scope in
    let ctx = ss |> foldl (fun ctx (v, _, e) -> fv_expr e (def_var v ctx)) ctx in
    let ctx = ctx |> fv_var v in
    let ctx = ctx |> pop_scope in
    ctx

  and fv_receiver ctx (x, v, b) =
    let ctx = ctx |> push_scope in
    let ctx = ctx |> fv_var v in
    let ctx = ctx |> def_var x in
    let ctx = fv_block b ctx in
    let ctx = ctx |> pop_scope in
    ctx

  and fv_expr e ctx =
    match e with
    | EAccess (v, _) -> ctx |> fv_var v
    | EUpdate (v0, _, v1) -> ctx |> fv_var v0 |> fv_var v1
    | ECall (v, vs) -> ctx |> fv_var v |> fv_vars vs
    | ECast (v, _) -> ctx |> fv_var v
    | EEmit (v0, v1) -> ctx |> fv_var v0 |> fv_var v1
    | EEnwrap (_, _, v) -> ctx |> fv_var v
    | EIf (v, b0, b1) -> ctx |> fv_var v |> fv_block b0 |> fv_block b1
    | EIs (_, _, v) -> ctx |> fv_var v
    | ELit _ -> ctx
    | ELoop b -> ctx |> fv_block b
    | EReceive v -> ctx |> fv_var v
    | EOn rs -> rs |> foldl fv_receiver ctx
    | ERecord vfs -> vfs |> foldl (fun ctx (_, v) -> ctx |> fv_var v) ctx
    | EUnwrap (_, _, v) -> ctx |> fv_var v
    | EReturn v -> ctx |> fv_var v
    | EBreak v -> ctx |> fv_var v
    | EContinue  -> ctx
    | EItem _ -> ctx
  in
  fv_block b ([vs], []) |> snd |> List.rev

and get_item xs (hir:hir) =
  match hir |> assoc_opt xs with
  | Some i -> i
  | None -> panic ("get_item: " ^ Pretty.path_to_str xs ^ " not found")
