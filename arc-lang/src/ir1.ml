open Utils
open Error

type ir1 = (path * item) list
and names = name list
and name = string
and paths = path list
and path = names
and arms = arm list
and arm = pattern * block
and params = param list
and param = name * ty
and index = int
and 't fields = 't field list
and 't field = name * 't
and 't record = 't fields * 't option
and variants = variant list
and variant = loc * name * tys
and block = stmts * expr
and generics = generic list
and generic = name

and stmts = stmt list
and stmt = SExpr of expr

and decorator = Ast.decorator

and async = bool

and items = item list
and item =
  | IClass       of loc * decorator * generics * bounds * instances
  | IDef         of loc * decorator * generics * params * ty * bounds * block
  | IExternDef   of loc * decorator * async * generics * tys * ty * bounds
  | IExternType  of loc * decorator * generics * bounds
  | IType        of loc * decorator * generics * ty * bounds
  | IVal         of loc * decorator * ty * block

and instances = instance list
and instance = generics * tys * bounds

and bounds = bound list
and bound = path * tys

and patterns = pattern list
and pattern =
  | PIgnore  of loc * ty
  | POr      of loc * ty * pattern * pattern
  | PRecord  of loc * ty * pattern record
  | PConst   of loc * ty * lit
  | PVar     of loc * ty * name
  | PUnwrap  of loc * ty * name * pattern

and tys = ty list
and ty =
  | TFunc      of tys * ty
  | TRecord    of ty
  | TEnum      of ty
  | TRowEmpty
  | TRowExtend of ty field * ty
  | TNominal   of path * tys
  | TGeneric   of name
  | TInverse   of ty
  | TVar       of name

and lit = Ast.lit

and exprs = expr list
and expr =
  | EAccess   of loc * ty * expr * name
  | EBreak    of loc * ty * expr
  | ECallExpr of loc * ty * expr * exprs
  | ECallItem of loc * ty * path * tys * exprs
  | ECast     of loc * ty * expr * ty
  | EContinue of loc * ty
  | EEnwrap   of loc * ty * name * expr
  | EItem     of loc * ty * path * tys
  | EVar      of loc * ty * name
  | ELit      of loc * ty * lit
  | ELoop     of loc * ty * block
  | EMatch    of loc * ty * expr * arms
  | EOn       of loc * ty * receivers
  | ERecord   of loc * ty * expr record
  | EReturn   of loc * ty * expr
  | EUpdate   of loc * ty * expr * name * expr
  | ESpawn    of loc * ty * path * tys * exprs

and receivers = receiver list
and receiver = pattern * expr * expr

let item_loc i =
  match i with
  | IExternDef   (i, _, _, _, _, _, _) -> i
  | IDef         (i, _, _, _, _, _, _) -> i
  | IVal         (i, _, _, _) -> i
  | IExternType  (i, _, _, _) -> i
  | IClass       (i, _, _, _, _) -> i
  | IType        (i, _, _, _, _) -> i

(* Create a nominal type *)
let rec nominal x gs = TNominal (["std"; x], gs)

(* Create an atomic type. Atomic types are defined in the standard library. *)
and atom x = nominal x []

(* Returns the parent path of a path *)
and parent xs = xs |> rev |> tl |> rev

let map_fields f fs = map (fun (x, v) -> (x, f v)) fs
let map_opt f v = match v with Some v -> Some (f v) | None -> None
let map_record f (fs, t) = (map_fields f fs, map_opt f t)

(* Map types *)
let rec tmap_item f i =
  match i with
  | IClass (loc, d, gs, bs, is) -> IClass (loc, d, gs, map (tmap_bound f) bs, map (tmap_inst f) is)
  | IDef (loc, d, gs, pts, ty, bs, b) -> IDef (loc, d, gs, map (tmap_param f) pts, f ty, map (tmap_bound f) bs, tmap_block f b)
  | IExternDef (loc, d, async, gs, tys, ty, bs) -> IExternDef (loc, d, async, gs, map f tys, f ty, map (tmap_bound f) bs)
  | IExternType (loc, d, gs, bs) -> IExternType (loc, d, gs, map (tmap_bound f) bs)
  | IType (loc, d, gs, ty, bs) -> IType (loc, d, gs, f ty, map (tmap_bound f) bs)
  | IVal (loc, d, ty, b) -> IVal (loc, d, f ty, tmap_block f b)

and tmap_bound f (x, ts) = (x, map f ts)
and tmap_inst f (gs, ts, bs) = (gs, map f ts, map (tmap_bound f) bs)

and tmap_expr f e =
  match e with
  | ELoop (loc, t, b) -> ELoop (loc, f t, tmap_block f b)
  | EEnwrap (loc, t, x, e) -> EEnwrap (loc, f t, x, tmap_expr f e)
  | EItem (loc, t, xs, ts) -> EItem (loc, f t, xs, map f ts)
  | ECallItem (loc, t, xs, ts, es) -> ECallItem (loc, f t, xs, map f ts, map (tmap_expr f) es)
  | EMatch (loc, t, e, arms) -> EMatch (loc, f t, tmap_expr f e, map (tmap_arm f) arms)
  | EAccess (loc, t, e, x) -> EAccess (loc, f t, tmap_expr f e, x)
  | EBreak (loc, t, e) -> EBreak (loc, f t, tmap_expr f e)
  | ECallExpr (loc, t, e, es) -> ECallExpr (loc, f t, tmap_expr f e, map (tmap_expr f) es)
  | ECast (loc, t0, e, t1) -> ECast (loc, f t0, tmap_expr f e, f t1)
  | EContinue (loc, t) -> EContinue (loc, f t)
  | ELit (loc, t, l) -> ELit (loc, f t, l)
  | EVar (loc, t, x) -> EVar (loc, f t, x)
  | EOn (loc, t, rs) -> EOn (loc, f t, map (tmap_receiver f) rs)
  | ERecord (loc, t, r) -> ERecord (loc, f t, map_record (tmap_expr f) r)
  | EReturn (loc, t, e) -> EReturn (loc, f t, tmap_expr f e)
  | EUpdate (loc, t, e0, x, e1) -> EUpdate (loc, f t, tmap_expr f e0, x, tmap_expr f e1)
  | ESpawn (loc, t, xs, ts, es) -> ESpawn (loc, f t, xs, map f ts, map (tmap_expr f) es)

and tmap_type f t =
  match t with
  | TFunc (ts, t) -> TFunc (map f ts, f t)
  | TRecord t -> TRecord (f t)
  | TEnum t -> TEnum (f t)
  | TRowEmpty -> TRowEmpty
  | TRowExtend ((x, t), r) -> TRowExtend ((x, f t), f r)
  | TNominal (xs, ts) -> TNominal (xs, map f ts)
  | TGeneric x -> TGeneric x
  | TInverse t -> TInverse (f t)
  | TVar x -> TVar x

and tmap_pat f p =
  match p with
  | PIgnore (loc, t) -> PIgnore (loc, f t)
  | POr (loc, t, p0, p1) -> POr (loc, f t, tmap_pat f p0, tmap_pat f p1)
  | PRecord (loc, t, r) -> PRecord (loc, f t, map_record (tmap_pat f) r)
  | PConst (loc, t, l) -> PConst (loc, f t, l)
  | PVar (loc, t, x) -> PVar (loc, f t, x)
  | PUnwrap (loc, t, x, p) -> PUnwrap (loc, f t, x, tmap_pat f p)

and tmap_arm f (p, b) = (tmap_pat f p, tmap_block f b)
and tmap_receiver f (p, e0, e1) = (tmap_pat f p, tmap_expr f e0, tmap_expr f e1)
and tmap_param f (x, t) = (x, f t)
and tmap_block f (es, e) = (map (tmap_stmt f) es, tmap_expr f e)
and tmap_stmt f s =
  match s with
  | SExpr e -> SExpr (tmap_expr f e)

(* Map expressions *)

and emap_item f i =
  match i with
  | IClass (loc, d, gs, bs, is) -> IClass (loc, d, gs, bs, is)
  | IDef (loc, d, gs, pts, ty, bs, b) -> IDef (loc, d, gs, pts, ty, bs, emap_block f b)
  | IExternDef (loc, d, async, gs, tys, ty, bs) -> IExternDef (loc, d, async, gs, tys, ty, bs)
  | IExternType (loc, d, gs, bs) -> IExternType (loc, d, gs, bs)
  | IType (loc, d, gs, ty, bs) -> IType (loc, d, gs, ty, bs)
  | IVal (loc, d, ty, b) -> IVal (loc, d, ty, emap_block f b)

and emap_expr f p =
  match p with
  | ELoop (loc, t, b) -> ELoop (loc, t, emap_block f b)
  | EEnwrap (loc, t, x, e) -> EEnwrap (loc, t, x, f e)
  | EItem (loc, t, xs, ts) -> EItem (loc, t, xs, ts)
  | ECallItem (loc, t, xs, ts, es) -> ECallItem (loc, t, xs, ts, map f es)
  | EMatch (loc, t, e, arms) -> EMatch (loc, t, f e, map (emap_arm f) arms)
  | EAccess (loc, t, e, x) -> EAccess (loc, t, f e, x)
  | EBreak (loc, t, e) -> EBreak (loc, t, f e)
  | ECallExpr (loc, t, e, es) -> ECallExpr (loc, t, f e, map f es)
  | ECast (loc, t0, e, t1) -> ECast (loc, t0, f e, t1)
  | EContinue (loc, t) -> EContinue (loc, t)
  | ELit (loc, t, l) -> ELit (loc, t, l)
  | EVar (loc, t, x) -> EVar (loc, t, x)
  | EOn (loc, t, rs) -> EOn (loc, t, map (emap_receiver f) rs)
  | ERecord (loc, t, r) -> ERecord (loc, t, map_record (emap_expr f) r)
  | EReturn (loc, t, e) -> EReturn (loc, t, f e)
  | EUpdate (loc, t, e0, x, e1) -> EUpdate (loc, t, f e0, x, f e1)
  | ESpawn (loc, t, xs, ts, es) -> ESpawn (loc, t, xs, ts, map (emap_expr f) es)

and emap_receiver f (p, e0, e1) = (p, f e0, f e1)
and emap_block f (es, e) = (map (emap_stmt f) es, f e)
and emap_stmt f s =
  match s with
  | SExpr e -> SExpr (f e)
and emap_arm f (p, b) = (p, emap_block f b)

(* Map patterns *)

let rec pmap_item f i =
  match i with
  | IClass (loc, d, gs, bs, is) -> IClass (loc, d, gs, bs, is)
  | IDef (loc, d, gs, pts, t, bs, b) -> IDef (loc, d, gs, pts, t, bs, emap_block f b)
  | IExternDef (loc, d, async, gs, ts, t, bs) -> IExternDef (loc, d, async, gs, ts, t, bs)
  | IExternType (loc, d, gs, bs) -> IExternType (loc, d, gs, bs)
  | IType (loc, d, gs, t, bs) -> IType (loc, d, gs, t, bs)
  | IVal (loc, d, t, b) -> IVal (loc, d, t, emap_block f b)

and pmap_expr f p =
  match p with
  | ELoop (loc, t, b) -> ELoop (loc, t, pmap_block f b)
  | EEnwrap (loc, t, x, e) -> EEnwrap (loc, t, x, pmap_expr f e)
  | EItem (loc, t, xs, ts) -> EItem (loc, t, xs, ts)
  | ECallItem (loc, t, xs, ts, es) -> ECallItem (loc, t, xs, ts, map (pmap_expr f) es)
  | EMatch (loc, t, e, arms) -> EMatch (loc, t, pmap_expr f e, map (pmap_arm f) arms)
  | EAccess (loc, t, e, x) -> EAccess (loc, t, pmap_expr f e, x)
  | EBreak (loc, t, e) -> EBreak (loc, t, pmap_expr f e)
  | ECallExpr (loc, t, e, es) -> ECallExpr (loc, t, pmap_expr f e, map (pmap_expr f) es)
  | ECast (loc, t0, e, t1) -> ECast (loc, t0, pmap_expr f e, t1)
  | EContinue (loc, t) -> EContinue (loc, t)
  | ELit (loc, t, l) -> ELit (loc, t, l)
  | EVar (loc, t, x) -> EVar (loc, t, x)
  | EOn (loc, t, rs) -> EOn (loc, t, map (pmap_receiver f) rs)
  | ERecord (loc, t, r) -> ERecord (loc, t, map_record (pmap_expr f) r)
  | EReturn (loc, t, e) -> EReturn (loc, t, pmap_expr f e)
  | EUpdate (loc, t, e0, x, e1) -> EUpdate (loc, t, pmap_expr f e0, x, pmap_expr f e1)
  | ESpawn (loc, t, xs, ts, es) -> ESpawn (loc, t, xs, ts, map (pmap_expr f) es)

and pmap_pat f p =
  match p with
  | PIgnore (loc, t) -> PIgnore (loc, t)
  | POr (loc, t, p0, p1) -> POr (loc, t, f p0, f p1)
  | PRecord (loc, t, r) -> PRecord (loc, t, map_record (pmap_pat f) r)
  | PConst (loc, t, l) -> PConst (loc, t, l)
  | PVar (loc, t, x) -> PVar (loc, t, x)
  | PUnwrap (loc, t, x, p) -> PUnwrap (loc, t, x, f p)

and pmap_arm f (p, b) = (f p, pmap_block f b)
and pmap_receiver f (p, e0, e1) = (f p, pmap_expr f e0, pmap_expr f e1)
and pmap_block f (es, e) = (map (pmap_stmt f) es, pmap_expr f e)
and pmap_stmt f s =
  match s with
  | SExpr e -> SExpr (pmap_expr f e)

(* Typeof *)

let typeof_expr e =
  match e with
  | EAccess (_, t, _, _) -> t
  | EUpdate (_, t, _, _, _) -> t
  | ECallExpr (_, t, _, _) -> t
  | ECallItem (_, t, _, _, _) -> t
  | ECast (_, t, _, _) -> t
  | EEnwrap (_, t, _, _) -> t
  | ELit (_, t, _) -> t
  | EVar (_, t, _) -> t
  | ELoop (_, t, _) -> t
  | EOn (_, t, _) -> t
  | ERecord (_, t, _) -> t
  | EReturn (_, t, _) -> t
  | EBreak (_, t, _) -> t
  | EContinue (_, t) -> t
  | EItem (_, t, _, _) -> t
  | EMatch (_, t, _, _) -> t
  | ESpawn (_, t, _, _, _) -> t

let typeof_pat p =
  match p with
  | PIgnore (_, t) -> t
  | POr (_, t, _, _) -> t
  | PRecord (_, t, _) -> t
  | PConst (_, t, _) -> t
  | PVar (_, t, _) -> t
  | PUnwrap (_, t, _, _) -> t

let typeof_block (_, e) = typeof_expr e

(* Convert an index of a tuple into a field of a record *)
let index_to_field i = Printf.sprintf "_%d" i

(* Convert indexes of a tuple to fields of a record *)
let indexes_to_fields is =
  is |> List.fold_left (fun (l, c) v -> ((index_to_field c, v)::l, c+1)) ([], 0)
     |> fst
     |> List.rev

(* Convert match arms to clauses. A clause has the following form: `(eqs, substs, expr)` where:
** - `eqs` are a set of equations of the form `(v, p)` where
**   - `v` is a variable
**   - `p` is a pattern match on the variable `v`
** - `substs` are a set of substitutions of the form `(v0, v1)` where
**   - `v0` is substituted for `v1` inside `expr`
** - `expr` is an expression which is evaluated if the clause succeeds
*)
let arms_to_clauses arms e =
  arms |> List.map (fun (p, b) -> ([(e, p)], [], b))

(* t is the tail, which could either be a Ir1.TVar or Ir1.TRowEmpty *)
let fields_to_rows t fs =
  fs |> List.fold_left (fun t f -> TRowExtend (f, t)) t

(* Converts a list [v0; v1; ..; vn] into [("_0", v0); ("_1", v1); ...; ("_n", vn)] *)
let indexes_to_rows t is =
  is |> indexes_to_fields |> fields_to_rows t

type ctx = {
  scopes: scopes;
  vars: name list
}
and scopes = scope list
and scope = name list

let bind_var v ctx =
  { ctx with scopes = (v::(hd ctx.scopes))::(tl ctx.scopes) }

let rec bind_pat ctx p =
  match p with
  | PIgnore _ -> ctx
  | POr (_, _, p0, p1) -> bind_pat (bind_pat ctx p1) p0
  | PRecord (_, _, (xps, p)) ->
      let ctx = foldl (fun ctx (_, p) -> bind_pat ctx p) ctx xps in
      begin match p with
      | Some p -> bind_pat ctx p
      | None -> ctx
      end
  | PConst _ -> ctx
  | PVar (_, _, x) -> bind_var x ctx
  | PUnwrap (_, _, _, ps) -> bind_pat ctx ps

let bound_vars ps =
  let ctx = { scopes = [[]]; vars = [] } in
  let ctx = ps |> foldl bind_pat ctx in
  ctx.scopes |> hd

(* Calculates the free variables of a block `b` parameterized by `vs` *)
let free_vars vs b =

  (* A variable is free if it is not bound in any scope *)
  let is_free v scopes = not (scopes |> exists (mem v)) in

  (* Convenience function *)
  let fv_var v ctx =
    if is_free v ctx.scopes then
      { ctx with vars = v::ctx.vars }
    else
      ctx
  in

  (* Push a new scope to the stack *)
  let push_scope ctx = { ctx with scopes = []::ctx.scopes } in

  (* Pop a scope off the stack *)
  let pop_scope ctx = { ctx with scopes = tl ctx.scopes } in

  (* Returns the list of free variables in a block *)
  let rec fv_block ctx (ss, e) =
    let ctx = push_scope ctx in
    let ctx = foldl fv_stmt ctx ss in
    let ctx = fv_expr ctx e in
    let ctx = pop_scope ctx in
    ctx

  and fv_stmt ctx s =
    match s with
    | SExpr e -> fv_expr ctx e

  (* Returns the list of free variables in a receiver *)
  and fv_receiver ctx (p, e0, e1) =
    let ctx = bind_pat ctx p in
    let ctx = fv_expr ctx e0 in
    let ctx = fv_expr ctx e1 in
    ctx

  and fv_arm ctx (p, b) =
    let ctx = push_scope ctx in
    let ctx = bind_pat ctx p in
    let ctx = fv_block ctx b in
    let ctx = pop_scope ctx in
    ctx

  (* Returns the list of free variables in an expression *)
  and fv_expr ctx e =
    match e with
    | EAccess (_, _, e, _) -> fv_expr ctx e
    | EUpdate (_, _, e0, _, e1) -> foldl fv_expr ctx [e0; e1]
    | ECallExpr (_, _, e, es) -> foldl fv_expr ctx (e::es)
    | ECallItem (_, _, _, _, es) -> foldl fv_expr ctx es
    | ECast (_, _, e, _) -> fv_expr ctx e
    | EEnwrap (_, _, _, e) -> fv_expr ctx e
    | ELit _ -> ctx
    | EVar (_, _, x) -> fv_var x ctx
    | ELoop (_, _, b) -> fv_block ctx b
    | EOn (_, _, rs) -> foldl fv_receiver ctx rs
    | ERecord (_, _, (xes, e)) ->
        let ctx = foldl (fun ctx (_, e) -> fv_expr ctx e) ctx xes in
        begin match e with
        | Some e -> fv_expr ctx e
        | None -> ctx
        end
    | EReturn (_, _, e) -> fv_expr ctx e
    | EBreak (_, _, e) -> fv_expr ctx e
    | EContinue _ -> ctx
    | EItem _ -> ctx
    | EMatch (_, _, e, arms) -> foldl fv_arm (fv_expr ctx e) arms
    | ESpawn (_, _, _, _, es) -> foldl fv_expr ctx es
  in
  let ctx = { scopes=[vs]; vars=[] } in
  let ctx = fv_block ctx b in
  ctx.vars |> List.rev

(* Given a path, returns the corresponding item from Ir1 *)
and get_item loc xs ir1 =
  match ir1 |> assoc_opt xs with
  | Some i -> i
  | None -> raise (Error.NamingError (loc, "get_item: " ^ Pretty.path_to_str xs ^ " not found"))
