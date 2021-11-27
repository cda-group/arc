open Utils
open Hir

module NameSet = Set.Make(struct type t = name let compare = compare end)
module NameMap = Map.Make(struct type t = name let compare = compare end)
module PathMap = Map.Make(struct type t = path let compare = compare end)

module Ctx = struct
  type t = {
    hir: Hir.hir;
    schemes: scheme PathMap.t; (* Inferred type schemes of items. *)
    definitions: (path * Hir.item) list;
    frames: frame list;
    next_type_uid: Gen.t;
    next_row_uid: Gen.t;
    subctx: subctx list;
    instances: ty list NameMap.t; (* Fully inferred instantiations *)
  }
  (* The frame of a polymorphic item which is currently being inferred *)
  and frame = {
    tsubst: (name * ty) list; (* Substitutions of type variables to types *)
    scopes: scope list;
    insts: ty list NameMap.t; (* Partially inferred instantiations *)
  }
  and scope = {
    vsubst: (name * ty) list; (* Substitutions of value variables to types *)
  }
  (* A scheme is a universally quantified type. Schemes can have both explicit
     and implicit type parameters. An explicit type parameter is declared
     explicitly while an implicit type parameter is inferred from the context.
     Explicit type parameters allow polymorphic recursion, while implicit do not. 

     fun id[T](x:T) { x }  # Explicit
     fun id(x) { x }       # Implicit

     1. To infer the type of a function definition, we:
     1.1. Check if we have already inferred the function type-scheme, if so we just return it.
     1.2. Otherwise, generate a fresh type variable `'a` for the function type.
     1.3. Push `'a` onto a stack of type scheme instantiations.
     1.4. Infer the type `t` of the function signature with respect to the function body. 
     1.5. Unify 'a against `t`.
     1.6. Generalize `'a` into a type scheme `sc`.
     1.7. Insert `sc` into a global context of type schemes.
     1.8. Pop `'a` from the stack of type scheme instantiations.
     1.8. Return the type scheme `sc`.

     2. When encountering a function reference, we look it up and instantiate it.
     2.1. This is needed to ensure that each function reference points to a concrete instance.

     2. When encountering the left-hand-side of an SSA operation
     2.1. Check if the right-hand-side expression is a value (function reference)
     2.1.1. If so, lookup the type scheme and bind the variable to it
     2.1.1. Else, infer the type of the right-hand-side and bind the variable to it

     2. When encountering an variable-operand of an SSA operation
     2.1. Check if the variable is bound to a type scheme
     2.1.1. If so, instantiate it
     2.1.2. Else, just use the type as it is

     Limitations:
     * Mutual recursion => Whichever function is processed first decides what becomes polymorphic.
     * Value restriction => 

     Value-level variables are always monomorphised
  *)
  and scheme =
    | SPoly of { t:ty; explicit_gs:name list; implicit_gs:name list; }
    | SMono of { t:ty; explicit_gs:name list; }

  and subctx =
    | CDef of ty
    | CTask of ty * ty
    | CLoop of ty

  let rec make hir = {
    hir = hir;
    schemes = PathMap.empty;
    definitions = [];
    frames = [];
    next_type_uid = Gen.make ();
    next_row_uid = Gen.make ();
    subctx = [];
    instances = NameMap.empty;
  }

  and return_ty ctx =
    match hd ctx.subctx with
    | CDef t -> t
    | _ -> panic "Tried to return outside of function"

  and input_event_ty ctx =
    match hd ctx.subctx with
    | CTask (t, _) -> t
    | _ -> panic "Tried to poll event outside of function"

  and output_event_ty ctx =
    match hd ctx.subctx with
    | CTask (_, t) -> t
    | _ -> panic "Tried to emit outside of task"

  and break_ty ctx =
    match hd ctx.subctx with
    | CLoop t -> t
    | _ -> panic "Tried to break outside of loop"

  and push_subctx subctx ctx = { ctx with subctx = subctx::ctx.subctx }

  and pop_subctx ctx = { ctx with subctx = tl ctx.subctx }

  and fresh_t ctx =
    let (i, next_type_uid) = ctx.next_type_uid |> Gen.fresh in
    let t = Hir.TVar (sprintf "a%d" i) in
    let ctx = { ctx with next_type_uid } in
    (t, ctx)

  and fresh_r ctx =
    let (i, next_row_uid) = ctx.next_row_uid |> Gen.fresh in
    let t = Hir.TVar (sprintf "a%d" i) in
    let ctx = { ctx with next_row_uid } in
    (t, ctx)

  and get_frame ctx = ctx.frames |> hd
  and get_scope ctx = (ctx |> get_frame).scopes |> hd
  and get_scopes ctx = (ctx |> get_frame).scopes
  and get_tsubst ctx = (ctx |> get_frame).tsubst
  and get_insts ctx = (ctx |> get_frame).insts
  and get_vsubst ctx = (ctx |> get_scope).vsubst

  and update_frame u ctx = match ctx.frames with
  | hd::tl -> { ctx with frames = (u hd)::tl }
  | [] -> unreachable ()

  and update_scope u ctx =
    match ctx.frames with
    | [] -> unreachable ()
    | f::frames ->
        match f.scopes with
        | [] -> unreachable ()
        | s::scopes -> { ctx with frames = { f with scopes = (u s)::scopes }::frames}
  and update_tsubst tsubst ctx = ctx |> update_frame (fun f -> { f with tsubst })
  and update_vsubst vsubst ctx = ctx |> update_scope (fun _ -> { vsubst })

  and bind_t x t ctx = ctx |> update_frame (fun f -> { f with tsubst = (x, t)::f.tsubst })
  and bind_v x t ctx =
    print_endline "Binding var";
    ctx |> update_scope (fun s -> { vsubst = (x, t)::s.vsubst })

  and find_t x ctx = ctx |> get_tsubst |> List.assoc_opt x
  and find_v x ctx = ctx |> get_scopes |> List.find_map (fun s -> s.vsubst |> List.assoc_opt x) |> Option.get
  and find_sc xs ctx = ctx.schemes |> PathMap.find xs
  and has_scheme xs ctx = ctx.schemes |> PathMap.exists (fun xs' _ -> xs' = xs)

  and push_frame ctx = { ctx with frames = {tsubst = []; scopes = []; insts = NameMap.empty}::ctx.frames } |> push_scope
  and pop_frame ctx =
    match ctx.frames with
    | hd::tl -> (hd.tsubst, hd.insts, { ctx with frames = tl })
    | _ -> unreachable ()
  and push_scope ctx = ctx |> update_frame (fun f -> { f with scopes = {vsubst = []}::f.scopes })
  and pop_scope ctx = ctx |> update_frame (fun f -> { f with scopes = tl f.scopes })

  and typeof x ctx = ctx |> find_v x

  and add_inst v ts ctx = ctx |> update_frame (fun f -> { f with insts = f.insts |> NameMap.add v ts})
  and add_scheme xs sc ctx = { ctx with schemes = ctx.schemes |> PathMap.add xs sc }
  and add_item xs i ctx = { ctx with definitions = (xs, i)::ctx.definitions}
end

let debug_schemes scs =
  Printf.printf "\nSchemes:\n";
  let ctx = Pretty.Ctx.brief in
  scs |> PathMap.iter (fun xs sc ->
    match sc with
    | Ctx.SPoly {t; explicit_gs; implicit_gs } ->
        Printf.printf "Poly: ";
        Pretty_hir.pr_path xs ctx;
        Printf.printf " => forall";
        Printf.printf "\n  explicit ";
        Printf.printf "[";
        Pretty.pr_sep ", " Pretty_hir.pr_name explicit_gs ctx;
        Printf.printf "]";
        Printf.printf "\n  implicit ";
        Printf.printf "[";
        Pretty.pr_sep ", " Pretty_hir.pr_name implicit_gs ctx;
        Printf.printf "]";
        Printf.printf "\n  . ";
        Pretty_hir.pr_type t ctx;
        Printf.printf "\n";
    | Ctx.SMono {t; explicit_gs } ->
        Printf.printf "Mono: ";
        Pretty_hir.pr_path xs ctx;
        Printf.printf " => forall ";
        Printf.printf "  explicit ";
        Printf.printf "[";
        Pretty.pr_sep ", " Pretty_hir.pr_name explicit_gs ctx;
        Printf.printf "]";
        Printf.printf "\n  . ";
        Pretty_hir.pr_type t ctx;
        Printf.printf "\n";
  )

let rec infer_each f l (ctx:Ctx.t) =
  l |> foldl f ctx

and ts_of_ps vs = vs |> map (fun (_, t) -> t)
and ts_of_vs vs ctx = vs |> map (fun v -> ctx |> Ctx.typeof v)
and fts_of_fvs vs ctx = vs |> map (fun (x, v) -> (x, ctx |> Ctx.typeof v))

and bind_params ps ctx =
  ps |> foldl (fun ctx (v, t) -> ctx |> Ctx.bind_v v t) ctx

and get_enum (ctx:Ctx.t) xs =
  match ctx.hir |> assoc xs with
  | Hir.IEnum (gs, xss) -> (gs, xss)
  | _ -> unreachable ()

and get_variant (ctx:Ctx.t) xs =
  match ctx.hir |> assoc xs with
  | Hir.IVariant t -> t
  | _ -> unreachable ()

(* Instantiate an enum_xs, unify its type parameters, and then unify it with type enum_t. *)
and instantiate_enum enum_xs enum_t ts0 (ctx:Ctx.t) =
  let (gs, _) = get_enum ctx enum_xs in
  let (ts1, s, ctx) = instantiate_generics gs ctx in
  let ctx = ctx |> try_unify_ts ts0 ts1 in
  let ctx = ctx |> unify enum_t (Hir.TNominal (enum_xs, ts1)) in
  (s, ctx)

(* Apply generic substitution to enum variant_xs and unify result with variant_t *)
and instantiate_variant variant_xs variant_t s (ctx:Ctx.t) =
  let t = get_variant ctx variant_xs in
  ctx |> unify variant_t (t |> instantiate s)

(* Instantiate generics `gs` into fresh type variables *)
and instantiate_generics gs ctx =
  let instantiate_generic g ctx =
    let (t, ctx) = ctx |> Ctx.fresh_t in
    ((g, t), ctx)
  in
  let (gts, ctx) = gs |> mapm instantiate_generic ctx in
  let ts = gts |> map snd in
  (ts, gts, ctx)

(* Apply type-variable substitution to type *)
and apply s t =
  let f t = apply s t in
  match t with
  | Hir.TFunc (ts, t) -> Hir.TFunc (map f ts, f t)
  | Hir.TRecord t -> Hir.TRecord (f t)
  | Hir.TRowEmpty -> Hir.TRowEmpty
  | Hir.TRowExtend ((x, t), r) -> Hir.TRowExtend ((x, f t), f r)
  | Hir.TNominal (xs, ts) -> Hir.TNominal (xs, map f ts)
  | Hir.TGeneric x -> Hir.TGeneric x
  | Hir.TVar x -> s |> get_or x t

(* Apply generic substitutions to type *)
and instantiate s t =
  let f t = instantiate s t in
  match t with
  | Hir.TFunc (ts, t) -> Hir.TFunc (map f ts, f t)
  | Hir.TRecord t -> Hir.TRecord (f t)
  | Hir.TRowEmpty -> Hir.TRowEmpty
  | Hir.TRowExtend ((x, t), r) -> Hir.TRowExtend ((x, f t), f r)
  | Hir.TNominal (xs, ts) -> Hir.TNominal (xs, map f ts)
  | Hir.TGeneric x -> s |> assoc x
  | Hir.TVar x -> Hir.TVar x

(* Generalise a type into a type scheme *)
and generalise s t =
  let f t = generalise s t in
  match t with
  | Hir.TFunc (ts, t) -> Hir.TFunc (map f ts, f t)
  | Hir.TRecord t -> Hir.TRecord (f t)
  | Hir.TRowEmpty -> Hir.TRowEmpty
  | Hir.TRowExtend ((x, t), r) -> Hir.TRowExtend ((x, f t), f r)
  | Hir.TNominal (xs, ts) -> Hir.TNominal (xs, map f ts)
  | Hir.TGeneric x -> Hir.TGeneric x
  | Hir.TVar x -> Hir.TGeneric (s |> assoc x)

(* Unifies two types *)
and unify t0 t1 (ctx:Ctx.t) =
  begin
    let ctx = Pretty.Ctx.brief in
    Printf.printf "Unifying: ";
    Pretty_hir.pr_type t0 ctx;
    Printf.printf " = ";
    Pretty_hir.pr_type t1 ctx;
    Printf.printf " \n";
  end;
  let s0 = ctx |> Ctx.get_tsubst in
  let (s1, ctx) = ctx |> mgu (apply s0 t0) (apply s0 t1) in
(*   Debug.debug_substitutions s1 ctx.hir; *)
  let ctx = ctx |> Ctx.update_tsubst (compose s1 s0) in
(*   Printf.printf "---------------------------------"; *)
  Debug.debug_substitutions (ctx |> Ctx.get_tsubst);
  Printf.printf "=================================\n\n";
  ctx

and try_unify_ts ts0 ts1 (ctx:Ctx.t) =
  if ts0 <> [] && ts1 <> [] then
    ctx |> unify_ts ts0 ts1
  else
    ctx

and unify_ts ts0 ts1 (ctx:Ctx.t) =
  zip ts0 ts1 |> foldl (fun ctx (t0, t1) -> unify t0 t1 ctx) ctx

and compose s0 s1 =
  (s1 |> map (fun (x, t) -> (x, apply s0 t))) @ s0

(* Returns a list of type variables occuring in a type *)
and tvs t =
  (* Returns the type variables occuring in t *)
  let rec tvs_of_t t acc =
    match t with
    | Hir.TFunc (ts, t) -> acc |> tvs_of_ts ts |> tvs_of_t t
    | Hir.TRecord t -> acc |> tvs_of_t t
    | Hir.TRowEmpty -> NameSet.empty
    | Hir.TRowExtend ((_, t), _) -> acc |> tvs_of_t t
    | Hir.TNominal (_, ts) -> acc |> tvs_of_ts ts
    | Hir.TGeneric _ -> NameSet.empty
    | Hir.TVar x -> acc |> NameSet.add x
  and tvs_of_ts ts acc = ts |> foldl (fun acc t -> (tvs_of_t t acc)) acc in
  tvs_of_t t NameSet.empty |> NameSet.elements

and mgus ts0 ts1 s (ctx:Ctx.t) =
  zip ts0 ts1 |> foldl (fun (s0, ctx) (t0, t1) ->
    let (s1, ctx) = ctx |> mgu (apply s0 t0) (apply s0 t1) in
    (compose s1 s0, ctx)
  ) (s, ctx)

and mgu t0 t1 ctx : ((Hir.name * Hir.ty) list * Ctx.t) =
  match t0, t1 with
  | Hir.TFunc (ts0, t0), Hir.TFunc (ts1, t1) ->
      ctx |> mgus (t0::ts0) (t1::ts1) []
  | Hir.TRecord t0, Hir.TRecord t1 ->
      ctx |> mgu t0 t1
  | Hir.TRowEmpty, Hir.TRowEmpty ->
      ([], ctx)
  | Hir.TRowExtend ((x0, t0), r0), (Hir.TRowExtend _ as r1) ->
      let (t1, r1, s, ctx) = ctx |> rewrite_row x0 r1 in
      ctx |> mgus [t0; r0] [t1; r1] s 
  | Hir.TNominal (xs0, ts0), Hir.TNominal (xs1, ts1) when xs0 = xs1 ->
      ctx |> mgus ts0 ts1 []
  | Hir.TGeneric x0, Hir.TGeneric x1 when x0 = x1 ->
      ([], ctx)
  | Hir.TVar x, t | t, Hir.TVar x ->
      if t = Hir.TVar x then
        ([], ctx)
      else if mem x (tvs t) then
        panic "Occurs check failure"
      else
        ([(x, t)], ctx)
  | _ ->
      let ctx = Pretty.Ctx.brief in
      Printf.printf "Oops... ";
      Pretty_hir.pr_type t0 ctx;
      Printf.printf " != ";
      Pretty_hir.pr_type t1 ctx;
      Printf.printf " \n";
      panic "Types do not unify"

and rewrite_row x0 r0 (ctx:Ctx.t) =
  match r0 with
  | Hir.TRowEmpty ->
      (* We've reached the end of the record, and it's already bound, so the new label cannot be inserted. *)
      panic (Printf.sprintf "label %s cannot be inserted" x0)
  | Hir.TRowExtend ((x1, t1), r1) when x0 = x1 ->
      (* We've found the label, so propagate it upwards *)
      (t1, r1, [], ctx)
  | Hir.TVar _ as r1 ->
      (* We've reached the end of the record, and it's not bound, so extend it and return the new tail *)
      let (t2, ctx) = ctx |> Ctx.fresh_t in
      let (r2, ctx) = ctx |> Ctx.fresh_r in
      let (s, ctx) = ctx |> mgu r1 (Hir.TRowExtend ((x0, t2), r2)) in
      (t2, r2, s, ctx)
  | Hir.TRowExtend ((x1, t1), r1) ->
      (* Otherwise, rewrite the tail *)
      let (t2, r2, s, ctx) = ctx |> rewrite_row x0 r1 in
      (t2, Hir.TRowExtend ((x1, t1), r2), s, ctx)
  | _ ->
      panic "Unexpected type"

let rec infer_hir hir = 
  let (ctx:Ctx.t) = Ctx.make hir in
  let ctx = hir |> foldl infer_item ctx in
  let hir = ctx.definitions |> map (fun (xs, i) -> (xs, Hir.smap_item (inst_ssa ctx.instances) i)) in
  hir |> rev

and inst_ssa s (v, t, e) =
  match e with
  | Hir.EItem (xs, _) ->
      let ts = s |> NameMap.find v in
      (v, t, Hir.EItem (xs, ts))
  | _ -> (v, t, Hir.smap_expr (inst_ssa s) e)

and infer_block (ss, v) (ctx:Ctx.t) =
  let ctx = ctx |> Ctx.push_scope in
  let ctx = ss |> foldl infer_ssa ctx in
  let t = ctx |> Ctx.typeof v in
  let ctx = ctx |> Ctx.pop_scope in
  (t, ctx)

and implicit_generics t =
  let tvs = tvs t in
  let (implicit_gs, _) = tvs |> foldl (fun (gs, i) _ -> ((sprintf "G%d" i)::gs, i+1)) ([], 0) in
  let s = zip tvs implicit_gs in
  (implicit_gs, s)

and infer_item (ctx:Ctx.t) (xs, i) : Ctx.t =
  if ctx |> Ctx.has_scheme xs then
    ctx
  else
    match i with
    | Hir.IClass _ -> todo ()
    | Hir.IInstance _ -> todo ()
    | Hir.IClassDecl _ -> todo ()
    | Hir.IInstanceDef _ -> todo ()
    | Hir.IVal (t0, b) ->
        let (t1, ctx) = infer_block b ctx in
        let _ctx = ctx |> unify t0 t1 in
        todo ()
    | Hir.IDef (explicit_gs, ps, t0, b) ->
        let ctx = ctx |> Ctx.push_frame in
        let ctx = ctx |> Ctx.push_subctx (Ctx.CDef t0) in

        (* Infer MGU of the function *)
        let ctx = ctx |> bind_params ps in
        let (t1, ctx) = ctx |> infer_block b in
        let ctx = ctx |> unify t0 t1 in

        let ctx = ctx |> Ctx.pop_subctx in
        let (s, insts, ctx) = ctx |> Ctx.pop_frame in

        (* Store instances in global context *)
        let insts = insts |> NameMap.map (map (apply s)) in
        let ctx = { ctx with instances = ctx.instances |> NameMap.add_seq (NameMap.to_seq insts) } in

        (* Apply MGU to all types in the function *)
        let (ps, t0, b) = tmap_func (apply s) (ps, t0, b) in

        (* Create the function type *)
        let t = Hir.TFunc (ps |> ts_of_ps, t0) in

        (* Get the implicit generics *)
        let (implicit_gs, s) = implicit_generics t in

        (* Replace all type variables with generics *)
        let t = t |> generalise s in
        let (ps, t0, b) = tmap_func (generalise s) (ps, t0, b) in

        (* Store the type scheme / generic item *)
        let sc = Ctx.SPoly { t; explicit_gs; implicit_gs } in
        let ctx = ctx |> Ctx.add_item xs (Hir.IDef (explicit_gs @ implicit_gs, ps, t0, b)) in
        let ctx = ctx |> Ctx.add_scheme xs sc in

        ctx
    | Hir.ITask (explicit_gs, ps, (xs0, ts0), (xs1, ts1), b) ->
        (* Create input/output event-types *)
        let ts = explicit_gs |> map (fun g -> Hir.TGeneric g) in
        let t0, t1 = (nominal xs0 ts), (nominal xs1 ts) in

        let ctx = ctx |> Ctx.push_frame in
        let ctx = ctx |> Ctx.push_subctx (Ctx.CTask (t0, t1)) in

        (* Infer MGU of the function *)
        let ctx = ctx |> bind_params ps in
        let (t2, ctx) = infer_block b ctx in
        let ctx = ctx |> unify (atom "unit") t2 in

        let ctx = ctx |> Ctx.pop_subctx in
        let (s, insts, ctx) = ctx |> Ctx.pop_frame in

        (* Store instances in global context *)
        let insts = insts |> NameMap.map (map (apply s)) in
        let ctx = { ctx with instances = ctx.instances |> NameMap.add_seq (NameMap.to_seq insts) } in

        (* Apply MGU to all types in the function *)
        let (ps, ts0, ts1, b) = tmap_task (apply s) (ps, ts0, ts1, b) in

        let ts0_streams = ts0 |> map (fun t -> Hir.TNominal (["Stream"], [t])) in
        let ts1_streams = ts1 |> map (fun t -> Hir.TNominal (["Stream"], [t])) in

        (* Create the task type *)
        let t = match ts1_streams with
        | [t3] ->
            Hir.TFunc (ts_of_ps ps, Hir.TFunc (ts0_streams, t3))
        | _ ->
            let t3 = ts1_streams |> Hir.indexes_to_rows Hir.TRowEmpty in
            Hir.TFunc (ts_of_ps ps, Hir.TFunc (ts0_streams, Hir.TRecord t3))
        in

        (* Get the implicit generics *)
        let (implicit_gs, s) = implicit_generics t in

        (* Replace all type variables with generics *)
        let t = t |> generalise s in
        let (ps, ts0, ts1, b) = tmap_task (generalise s) (ps, ts0, ts1, b) in

        (* Store the type scheme / generic item *)
        let sc = Ctx.SPoly { t; explicit_gs; implicit_gs } in
        let ctx = ctx |> Ctx.add_item xs (Hir.ITask (explicit_gs, ps, (xs0, ts0), (xs1, ts1), b)) in
        let ctx = ctx |> Ctx.add_scheme xs sc in
        ctx
    | Hir.IEnum _
    | Hir.IExternDef _
    | Hir.IExternType _
    | Hir.ITypeAlias _
    | Hir.IVariant _ -> ctx |> Ctx.add_item xs i

and infer_lit t l (ctx:Ctx.t) =
  match l with
  | Ast.LInt (_, _suffix) ->
      ctx |> unify t (atom "i32")
  | Ast.LFloat (_, _suffix) ->
      ctx |> unify t (atom "f32")
  | Ast.LBool _ ->
      ctx |> unify t (atom "bool")
  | Ast.LString _ ->
      ctx |> unify t (atom "string")
  | Ast.LUnit ->
      ctx |> unify t (atom "unit")
  | Ast.LChar _ ->
      ctx |> unify t (atom "char")

and infer_unop t0 op v1 (ctx:Ctx.t) =
  let typeof v = ctx |> Ctx.typeof v in
  match op with
  | Ast.UNeg ->
      ctx |> unify t0 (atom "bool")
          |> unify t0 (typeof v1)
  | Ast.UNot ->
      ctx |> unify t0 (atom "bool")
          |> unify t0 (typeof v1)

and infer_ssa (ctx:Ctx.t) (v0, t0, e0) =
  let ctx = infer_ssa_rhs ctx (v0, t0, e0) in
  let ctx = ctx |> Ctx.bind_v v0 t0 in
  ctx

and infer_ssa_rhs ctx (v0, t0, e0) =
  let typeof v = ctx |> Ctx.typeof v in
  match e0 with
  | Hir.EAccess (v1, x) ->
      let (t2, ctx) = ctx |> Ctx.fresh_t in
      let t3 = Hir.TRowExtend ((x, t0), t2) in
      ctx |> unify (Hir.TRecord t3) (typeof v1)
  | Hir.EAfter (v1, b)
  | Hir.EEvery (v1, b) ->
      let (t2, ctx) = infer_block b ctx in
      ctx |> unify (typeof v1) (atom "i32")
          |> unify t2 (atom "unit")
          |> unify (typeof v0) t2
  | Hir.ECall (v1, vs) ->
      ctx |> unify (typeof v1) (Hir.TFunc (ts_of_vs vs ctx, t0))
  | Hir.ECast (v1, t2) ->
      ctx |> unify t0 (typeof v1)
          |> unify t0 t2
  | Hir.EEmit v1 ->
      ctx |> unify t0 (atom "unit")
          |> unify (typeof v1) (ctx |> Ctx.output_event_ty)
  | Hir.EEq (v1, v2) ->
      ctx |> unify t0 (atom "bool")
          |> unify (typeof v1) (typeof v2)
  | Hir.EIf (v1, b0, b1) ->
      let (t2, ctx) = infer_block b0 ctx in
      let (t3, ctx) = infer_block b1 ctx in
      ctx |> unify t0 t2
          |> unify t0 t3
          |> unify (atom "bool") (typeof v1)
  | Hir.ELit l ->
      infer_lit t0 l ctx
  | Hir.ELoop b ->
      let (t1, ctx) = infer_block b ctx in
      ctx |> unify t0 (atom "unit")
          |> unify t1 (atom "unit")
  | Hir.EReceive -> ctx
  | Hir.ERecord fvs ->
      let fts = fts_of_fvs fvs ctx in
      let t1 = fts |> Hir.fields_to_rows Hir.TRowEmpty in
      ctx |> unify t0 (Hir.TRecord t1)
  | Hir.EReturn v1 ->
      ctx |> unify t0 (atom "unit")
          |> unify (typeof v1) (ctx |> Ctx.return_ty)
  | Hir.EBreak v1 ->
      ctx |> unify t0 (atom "unit")
          |> unify (typeof v1) (ctx |> Ctx.break_ty)
  | Hir.EContinue ->
      ctx |> unify t0 (atom "unit")
  | Hir.EEnwrap (xs, ts1, v1) ->
      let (s, ctx) = ctx |> instantiate_enum (xs |> parent) t0 ts1 in
      instantiate_variant xs (typeof v1) s ctx
  | Hir.EUnwrap (xs, ts1, v1) ->
      let (s, ctx) = ctx |> instantiate_enum (xs |> parent) (typeof v1) ts1 in
      instantiate_variant xs t0 s ctx
  | Hir.EIs (xs, ts1, v1) ->
      let (_, ctx) = ctx |> instantiate_enum (xs |> parent) (typeof v1) ts1 in
      unify t0 (atom "bool") ctx
  (* Every item reference must point to a concrete monomorphised instance of
     the referenced item. Therefore we must instantiate the item directly when
     referencing it. This is similar to how it works in Rust where function
     pointers must be monomorphic.

     fun duplicate(x0) { (x0, x0) }

     fun does_not_work() {
         val x1 = duplicate;  # OK
         val x2 = 1;          # OK
         val x3 = x1(x2, x2); # OK
         val x4 = "1";        # OK
         val x5 = x1(x4, x4); # Error, type mismatch
     }

     It's sensible, function pointers are values. In the general case we cannot
     know what a value is unless we evaluate the program. Thus, it would not be
     possible to know what a function pointer monomorphises to at compiletime.
     The following is possible however.

     fun does_work() {
         val x1 = duplicate;  # OK
         val x2 = 1;          # OK
         val x3 = x1(x2, x2); # OK
         val x4 = duplicate;  # OK
         val x5 = "1";        # OK
         val x6 = x4(x5, x5); # OK
     }
  *)
  | Hir.EItem (xs, ts0) ->
      let i = ctx.hir |> assoc xs in
      let ctx = infer_item ctx (xs, i) in
      let sc = ctx |> Ctx.find_sc xs in
      match sc with
      (* If we have already inferenced this item then we can just instantiate
         its type scheme *)
      | Ctx.SPoly sc ->
          debug_schemes ctx.schemes;
          let (ts1, s0, ctx) = ctx |> instantiate_generics sc.explicit_gs in
          let (ts2, s1, ctx) = ctx |> instantiate_generics sc.implicit_gs in
          let ctx = ctx |> try_unify_ts ts0 ts1 in
          let t1 = sc.t |> instantiate (s0 @ s1) in
          let ctx = ctx |> unify t0 t1 in
          let ctx = ctx |> Ctx.add_inst v0 (ts1 @ ts2) in
          ctx
      (* If we are currently inferencing this item, then we must reuse its
         implicit types to avoid polymorphic infinite recursion. *)
      | Ctx.SMono sc ->
          let (ts1, s, ctx) = ctx |> instantiate_generics sc.explicit_gs in
          let ctx = ctx |> try_unify_ts ts0 ts1 in
          let t1 = sc.t |> instantiate s in
          let ctx = ctx |> unify t0 t1 in
          let ctx = ctx |> Ctx.add_inst v0 ts1 in
          ctx

