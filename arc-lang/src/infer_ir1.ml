open Utils
open Ir1

module NameSet = Set.Make(struct type t = name let compare = compare end)
module PathMap = Map.Make(struct type t = path let compare = compare end)

module Ctx = struct
  type t = {
    ir1: ir1;
    schemes: scheme PathMap.t; (* Inferred type schemes of items. *)
    items: (path * item) list;
    istack: iscope list;
    next_type_uid: Gen.t;
    next_row_uid: Gen.t;
    cstack: cscope list;
  }
  (* The iscope of a polymorphic item which is currently being inferred *)
  and iscope = {
    tsubst: (name * ty) list; (* Substitutions of type variables to types *)
    vstack: vscope list;
  }
  and vscope = {
    vsubst: (name * ty) list; (* Substitutions of value variables to types *)
  }
  and cscope = (* Control-flow scope *)
    | CDef of { return_ty: ty }
    | CTask
    | CLoop of { break_ty: ty }

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
     * Value restriction => Value-level variables are always monomorphised
  *)
  and scheme =
    | SPoly of { t_fun:ty; explicit_gs:name list; implicit_gs:name list; }
    | SMono of { t_fun:ty; explicit_gs:name list; }

  let rec make ir1 = {
    ir1 = ir1;
    schemes = PathMap.empty;
    items = [];
    istack = [];
    next_type_uid = Gen.make ();
    next_row_uid = Gen.make ();
    cstack = [];
  }

  and return_ty ctx =
    ctx.cstack
      |> List.find_map (function CDef s -> Some s.return_ty | _ -> None)
      |> Option.get

  and break_ty loc ctx =
    match hd ctx.cstack with
    | CLoop hd -> hd.break_ty
    | _ -> raise (Error.TypingError (loc, "Tried to break outside of loop"))

  and push_subctx cscope ctx = { ctx with cstack = cscope::ctx.cstack }

  and pop_subctx ctx = { ctx with cstack = tl ctx.cstack }

  and fresh_t ctx =
    let (i, next_type_uid) = ctx.next_type_uid |> Gen.fresh in
    let t = TVar (sprintf "g%d" i) in
    let ctx = { ctx with next_type_uid } in
    (t, ctx)

  and fresh_r ctx =
    let (i, next_row_uid) = ctx.next_row_uid |> Gen.fresh in
    let t = TVar (sprintf "r%d" i) in
    let ctx = { ctx with next_row_uid } in
    (t, ctx)

  and get_iscope ctx = ctx.istack |> hd
  and get_vscope ctx = (ctx |> get_iscope).vstack |> hd
  and get_vstack ctx = (ctx |> get_iscope).vstack
  and get_tsubst ctx = (ctx |> get_iscope).tsubst
  and get_vsubst ctx = (ctx |> get_vscope).vsubst

  and update_iscope f ctx =
    { ctx with istack = (f (hd ctx.istack))::(tl ctx.istack) }

  and update_vscope f ctx =
    let f iscope = { iscope with vstack = (f (hd iscope.vstack))::(tl iscope.vstack) } in
    ctx |> update_iscope f

  and update_tsubst tsubst ctx =
    ctx |> update_iscope (fun iscope -> { iscope with tsubst })

  and update_vsubst vsubst ctx =
    ctx |> update_vscope (fun _ -> { vsubst })

  and bind_var x t ctx =
    ctx |> update_vscope (fun s -> { vsubst = (x, t)::s.vsubst })

  and typeof_var loc x ctx =
    match ctx |> get_vstack |> find_map (fun s -> s.vsubst |> assoc_opt x) with
    | Some t -> t
    | None -> raise (Error.TypingError (loc, "Variable not bound: " ^ x))

  and get_scheme xs ctx =
    ctx.schemes |> PathMap.find xs

  and has_scheme xs ctx =
    ctx.schemes |> PathMap.exists (fun xs' _ -> xs' = xs)

  and push_iscope ctx =
    { ctx with istack = { tsubst = []; vstack = []; }::ctx.istack } |> push_vscope

  and pop_iscope ctx =
    let iscope = hd ctx.istack in
    let istack = tl ctx.istack in
    (iscope.tsubst, { ctx with istack })

  and push_vscope ctx =
    ctx |> update_iscope (fun f -> { f with vstack = { vsubst = [] }::f.vstack })

  and pop_vscope ctx =
    ctx |> update_iscope (fun f -> { f with vstack = tl f.vstack })

  and add_scheme xs sc ctx =
    { ctx with schemes = ctx.schemes |> PathMap.add xs sc }

  and add_item xs i ctx =
    { ctx with items = (xs, i)::ctx.items }
end

let infer_opt f v ctx =
  match v with
  | Some v ->
      let (v, ctx) = f v ctx in
      (Some v, ctx)
  | None ->
      (None, ctx)

let debug_schemes scs =
  Printf.printf "Schemes:\n";
  let ctx = Print.Ctx.make in
  scs |> PathMap.iter (fun xs sc ->
    match sc with
    | Ctx.SPoly {t_fun; explicit_gs; implicit_gs } ->
        Printf.printf "Poly: ";
        Print.pr_path xs ctx;
        Printf.printf " => forall";
        Printf.printf "\n  explicit ";
        Printf.printf "[";
        Print.pr_sep ", " Print.pr_name explicit_gs ctx;
        Printf.printf "]";
        Printf.printf "\n  implicit ";
        Printf.printf "[";
        Print.pr_sep ", " Print.pr_name implicit_gs ctx;
        Printf.printf "]";
        Printf.printf "\n  . ";
        Print.Ir1.pr_type t_fun ctx;
        Printf.printf "\n";
    | Ctx.SMono {t_fun; explicit_gs } ->
        Printf.printf "Mono: ";
        Print.pr_path xs ctx;
        Printf.printf " => forall ";
        Printf.printf "  explicit ";
        Printf.printf "[";
        Print.pr_sep ", " Print.pr_name explicit_gs ctx;
        Printf.printf "]";
        Printf.printf "\n  . ";
        Print.Ir1.pr_type t_fun ctx;
        Printf.printf "\n";
  )

let rec ts_of_xts xts = xts |> map (fun (_, t) -> t)
and ts_of_es es = es |> map (fun e -> typeof_expr e)
and xts_of_xes xes = xes |> map (fun (x, e) -> (x, typeof_expr e))
and xts_of_xps xps = xps |> map (fun (x, e) -> (x, typeof_pat e))

and bind_params ps ctx =
  ps |> foldl (fun ctx (v, t) -> ctx |> Ctx.bind_var v t) ctx

(* Instantiate an enum_xs, unify its type parameters, and then unify it with type enum_t. *)
and instantiate_generic g ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  if !Args.verbose then begin
    Printf.printf "Instantiating generic: ";
    Print.Ir1.pr_name g Print.Ctx.make;
    Printf.printf " => ";
    Print.Ir1.pr_type t Print.Ctx.make;
  end;
  ((g, t), ctx)

(* Instantiate generics `gs` into fresh type variables *)
and instantiate_generics gs ctx =
  let (s, ctx) = gs |> mapm instantiate_generic ctx in
  let ts = s |> map snd in
  (ts, s, ctx)

(* Apply type-variable substitution to type *)
and apply s t =
  let f t = apply s t in
  match t with
  | TFunc (ts, t) -> TFunc (map f ts, f t)
  | TRecord t -> TRecord (f t)
  | TEnum t -> TEnum (f t)
  | TRowEmpty -> TRowEmpty
  | TRowExtend ((x, t), r) -> TRowExtend ((x, f t), f r)
  | TNominal (xs, ts) -> TNominal (xs, map f ts)
  | TGeneric x -> TGeneric x
  | TVar x -> s |> get_or x t
  | TInverse t -> TInverse (f t)

(* Apply generic substitutions to type *)
and instantiate s t =
  let f t = instantiate s t in
  match t with
  | TFunc (ts, t) -> TFunc (map f ts, f t)
  | TRecord t -> TRecord (f t)
  | TEnum t -> TEnum (f t)
  | TRowEmpty -> TRowEmpty
  | TRowExtend ((x, t), r) -> TRowExtend ((x, f t), f r)
  | TNominal (xs, ts) -> TNominal (xs, map f ts)
  | TGeneric x -> s |> assoc x
  | TVar x -> TVar x
  | TInverse t -> TInverse (f t)

(* Generalise a type into a type scheme *)
and generalise s t =
  let f t = generalise s t in
  match t with
  | TFunc (ts, t) -> TFunc (map f ts, f t)
  | TRecord t -> TRecord (f t)
  | TEnum t -> TEnum (f t)
  | TRowEmpty -> TRowEmpty
  | TRowExtend ((x, t), r) -> TRowExtend ((x, f t), f r)
  | TNominal (xs, ts) -> TNominal (xs, map f ts)
  | TGeneric x -> TGeneric x
  | TVar x -> TGeneric (s |> get x)
  | TInverse t -> TInverse (f t)

and debug_substitution (x0, x1) ctx =
  Printf.printf "  '";
  Print.pr_name x0 ctx;
  Printf.printf " -> ";
  Print.Ir1.pr_type x1 ctx;
  print_endline ""

and debug_substitutions s =
  Printf.printf "Substitutions:\n";
  Print.pr_sep "" debug_substitution s Print.Ctx.make;
  Printf.printf "\n";

(* Unifies two types *)
and unify loc t t1 ctx =
  if !Args.verbose then
    begin
      let ctx = Print.Ctx.make in
      Printf.printf "\n";
      Printf.printf "Unifying:  (";
      Print.Ir1.pr_type t ctx;
      Printf.printf ")  with  (";
      Print.Ir1.pr_type t1 ctx;
      Printf.printf ")\n";
    end;
  let s0 = ctx |> Ctx.get_tsubst in
  let (s1, ctx) = ctx |> mgu loc (apply s0 t) (apply s0 t1) in
  let s2 = (compose s1 s0) in
  let ctx = ctx |> Ctx.update_tsubst s2 in
  if !Args.verbose then begin
    Printf.printf "----";
    debug_substitutions s2;
  end;
  ctx

and unify_ts loc ts0 ts1 ctx =
  if ts0 <> [] && ts1 <> [] then
    zip ts0 ts1 |> foldl (fun ctx (t, t1) -> unify loc t t1 ctx) ctx
  else
    ctx

and compose s0 s1 =
  (s1 |> map (fun (x, t) -> (x, apply s0 t))) @ s0

(* Returns a list of type variables occuring in a type *)
and tvs t =
  (* Returns the type variables occuring in t *)
  let rec tvs acc t =
    match t with
    | TFunc (ts, t) -> (t::ts) |> foldl tvs acc
    | TRecord t -> tvs acc t
    | TEnum t -> tvs acc t
    | TRowEmpty -> acc
    | TRowExtend ((_, t), r) -> [t; r] |> foldl tvs acc
    | TNominal (_, ts) -> ts |> foldl tvs acc
    | TGeneric _ -> acc
    | TVar x -> acc |> NameSet.add x
    | TInverse t -> t |> tvs acc
  in
  let xs = t |> tvs NameSet.empty |> NameSet.elements in
  if !Args.verbose then begin
    Printf.printf "\n";
    Print.Ir1.pr_type t Print.Ctx.make;
    Printf.printf " has freevars [%s]" (xs |> String.concat ", ");
    Printf.printf "\n";
  end;
  xs

and mgus loc ts0 ts1 s ctx =
  if !Args.verbose then begin
    Printf.printf "\n[";
    Print.Ir1.pr_types ts0 Print.Ctx.make;
    Printf.printf "] == [";
    Print.Ir1.pr_types ts1 Print.Ctx.make;
    Printf.printf "]\n";
  end;
  zip ts0 ts1 |> foldl (fun (s0, ctx) (t, t1) ->
    let (s1, ctx) = ctx |> mgu loc (apply s0 t) (apply s0 t1) in
    (compose s1 s0, ctx)
  ) (s, ctx)

and mgu loc t0 t1 ctx =
  match t0, t1 with
  | TFunc (ts0, t0), TFunc (ts1, t1) ->
      ctx |> mgus loc (ts0 @ [t0]) (ts1 @ [t1]) []
  | TRecord t0, TRecord t1 ->
      ctx |> mgu loc t0 t1
  | TEnum t0, TEnum t1 ->
      ctx |> mgu loc t0 t1
  | TRowEmpty, TRowEmpty ->
      ([], ctx)
  | TRowExtend ((x0, t0), r0), (TRowExtend _ as r1) ->
      let (t1, r1, s, ctx) = ctx |> rewrite_row loc x0 r1 in
      ctx |> mgus loc [t0; r0] [t1; r1] s
  | TNominal (xs0, ts0), TNominal (xs1, ts1) when xs0 = xs1 ->
      ctx |> mgus loc ts0 ts1 []
  | TGeneric x0, TGeneric x1 when x0 = x1 ->
      ([], ctx)
  | TVar x, t | t, TVar x ->
      if t = TVar x then
        ([], ctx)
      else if mem x (tvs t) then
        panic "[ICE]: Occurs check failure"
      else
        ([(x, t)], ctx)
  | TInverse TNominal (["std"; "PushChan"], [t]), TNominal (["std"; "PullChan"], [t1]) ->
      ctx |> mgu loc t t1
  | TNominal (["std"; "PullChan"], [t]), TInverse TNominal (["std"; "PushChan"], [t1]) ->
      ctx |> mgu loc t t1
  | TInverse t, TInverse t1 ->
      ctx |> mgu loc t t1
  | _ ->
      Printf.printf "Oops... ";
      Print.Ir1.pr_type t0 Print.Ctx.make;
      Printf.printf " != ";
      Print.Ir1.pr_type t1 Print.Ctx.make;
      Printf.printf " \n";
      raise (Error.TypingError (loc, "Types do not unify\n"))

(* Returns a tuple (t, r, s, ctx) where
 * - t is the type of the field with name x0 in record r0
 * - r is the record with the field removed
 *)
and rewrite_row loc x0 r0 ctx =
  let rec rewrite_row r0 ctx =
    match r0 with
    | TRowEmpty ->
        (* We've reached the end of the record, and it's already bound, so the new label cannot be inserted. *)
        raise (Error.TypingError (loc, Printf.sprintf "Can't insert %s into an empty row" x0))
    | TRowExtend ((x1, t1), r1) ->
        if x0 = x1 then
          (* We've found the label, so propagate it upwards *)
          (t1, r1, [], ctx)
        else
          (* Otherwise, keep searching and update the tail of this row *)
          let (t2, r2, s, ctx) = ctx |> rewrite_row r1 in
          (t2, TRowExtend ((x1, t1), r2), s, ctx)
    | TVar _ as r1 ->
        (* We've reached the end of the record, and it's not bound, so extend it and return the new tail *)
        let (t2, ctx) = ctx |> Ctx.fresh_t in
        let (r2, ctx) = ctx |> Ctx.fresh_r in
        let (s, ctx) = ctx |> mgu loc r1 (TRowExtend ((x0, t2), r2)) in
        (t2, r2, s, ctx)
    | _ ->
        raise (Error.TypingError (loc, "Cannot unify a type with a row"))
  in
  rewrite_row r0 ctx

(* Returns a new IR1 where all types are inferred *)
let rec infer_ir1 ir1 =
  let ctx = Ctx.make ir1 in
  let ctx = ir1 |> foldl (fun ctx xsi -> infer_item xsi ctx) ctx in
  ctx.items |> rev

and infer_block (ss, e) ctx =
  let ctx = ctx |> Ctx.push_vscope in
  let (es, ctx) = ss |> mapm infer_stmt ctx in
  let (e, ctx) = ctx |> infer_expr e in
  let ctx = ctx |> Ctx.pop_vscope in
  ((es, e), ctx)

and infer_stmt s ctx =
  match s with
  | SExpr e ->
      let (e, ctx) = ctx |> infer_expr e in
      (SExpr e, ctx)

(* Create generics for all type variables occurring in a type *)
and implicit_generics t =
  let tvs = tvs t in
  let (implicit_gs, _) = tvs |> foldl (fun (gs, i) _ -> ((sprintf "G%d" i)::gs, i+1)) ([], 0) in
  let s = zip tvs implicit_gs in
  (implicit_gs, s)

and infer_item (xs, i) ctx =
  if ctx |> Ctx.has_scheme xs then
    ctx
  else
    match i with
    | IDef (loc, a, explicit_gs, ps, t, _bs, b) ->
        if !Args.verbose then Print.Ir1.pr_item (xs, i) Print.Ctx.make;
        
        let ctx = ctx |> Ctx.push_iscope in
        let ctx = ctx |> Ctx.push_subctx (Ctx.CDef {return_ty=t}) in

        (* Infer MGU of the function *)
        let ctx = ctx |> bind_params ps in
        let (b, ctx) = ctx |> infer_block b in
        let ctx = ctx |> unify loc t (typeof_block b) in

        let ctx = ctx |> Ctx.pop_subctx in
        let (s, ctx) = ctx |> Ctx.pop_iscope in

        if !Args.verbose then debug_substitutions s;

        (* Apply MGU to all types in the function *)
        let f = apply s in
        let ps = map (tmap_param f) ps in
        let t = f t in
        let b = tmap_block f b in

        (* Create the function type *)
        let t_fun = TFunc (ts_of_xts ps, t) in

        (* Get the implicit generics *)
        let (implicit_gs, s) = implicit_generics t_fun in

        (* Create the function *)
        let i = IDef (loc, a, explicit_gs @ implicit_gs, ps, t, _bs, b) in

        (* Generalise the function*)
        let f = generalise s in
        let t_fun = f t_fun in
        let i = tmap_item f i in

        (* Store the type scheme / generic function *)
        let sc = Ctx.SPoly { t_fun; explicit_gs; implicit_gs } in
        let ctx = ctx |> Ctx.add_item xs i in
        let ctx = ctx |> Ctx.add_scheme xs sc in

        ctx
    | IExternDef (_, _, _, explicit_gs, ts, t, _bs) ->
        let t_fun = TFunc (ts, t) in
        let ctx = ctx |> Ctx.add_scheme xs (Ctx.SPoly { t_fun; explicit_gs; implicit_gs = [] }) in
        let ctx = ctx |> Ctx.add_item xs i in
        ctx
    | IClass _ | IVal _ -> todo ()
    | IExternType _ | IType _ -> ctx |> Ctx.add_item xs i

and infer_lit loc t l ctx =
  match l with
  | Ast.LInt (_, _, None) ->
      ctx |> unify loc t (atom "i32")
  | Ast.LInt (_, _, Some s) ->
      ctx |> unify loc t (atom s)
  | Ast.LFloat (_, _, None) ->
      ctx |> unify loc t (atom "f32")
  | Ast.LFloat (_, _, Some s) ->
      ctx |> unify loc t (atom s)
  | Ast.LBool _ ->
      ctx |> unify loc t (atom "bool")
  | Ast.LString _ ->
      ctx |> unify loc t (atom "str")
  | Ast.LUnit _ ->
      ctx |> unify loc t (atom "unit")
  | Ast.LChar _ ->
      ctx |> unify loc t (atom "char")

and infer_expr e ctx =
  match e with
  | EAccess (loc, t, e1, x) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let (r2, ctx) = ctx |> Ctx.fresh_r in
      let ctx = ctx |> unify loc (TRecord (TRowExtend ((x, t), r2))) (typeof_expr e1) in
      (EAccess (loc, t, e1, x), ctx)
  | EUpdate (loc, t, e1, x, e2) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let (e2, ctx) = ctx |> infer_expr e2 in
      let (r2, ctx) = ctx |> Ctx.fresh_r in
      let ctx = ctx |> unify loc t (atom "unit") in
      let ctx = ctx |> unify loc (TRecord (TRowExtend ((x, (typeof_expr e2)), r2))) (typeof_expr e1) in
      (EUpdate (loc, t, e1, x, e2), ctx)
  | ERecord (loc, t, (xes, e)) ->
      let (xes, ctx) = xes |> mapm infer_expr_field ctx in
      let (e, ctx) = ctx |> infer_opt infer_expr e in
      let xts = xts_of_xes xes in
      let (r, ctx) = match e with
        | None -> (TRowEmpty, ctx)
        | Some e ->
            let (r, ctx) = ctx |> Ctx.fresh_r in
            let ctx = ctx |> unify loc (TRecord r) (typeof_expr e) in
            (r, ctx)
      in
      let ctx = ctx |> unify loc t (TRecord (xts |> fields_to_rows r)) in
      (ERecord (loc, t, (xes, e)), ctx)
  | EEnwrap (loc, t, x, e1) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let ctx = ctx |> unify loc t (TEnum ([(x, typeof_expr e1)] |> fields_to_rows TRowEmpty)) in
      (EEnwrap (loc, t, x, e1), ctx)
  | ECast (loc, t, e1, t2) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let ctx = ctx |> unify loc t (typeof_expr e1) in
      let ctx = ctx |> unify loc t t2 in
      (ECast (loc, t, e1, t2), ctx)
  | EOn _ ->
      todo ()
  | ELit (loc, t, l) ->
      let ctx = ctx |> infer_lit loc t l in
      (ELit (loc, t, l), ctx)
  | ELoop (loc, t, b) ->
      let ctx = ctx |> Ctx.push_subctx (Ctx.CLoop { break_ty=t }) in
      let (b, ctx) = ctx |> infer_block b in
      let ctx = ctx |> unify loc (typeof_block b) (atom "unit") in
      let ctx = ctx |> Ctx.pop_subctx in
      (ELoop (loc, t, b), ctx)
  | EReturn (loc, t, e1) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let ctx = ctx |> unify loc (typeof_expr e1) (ctx |> Ctx.return_ty) in
      (EReturn (loc, t, e1), ctx)
  | EBreak (loc, t, e1) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let ctx = ctx |> unify loc (typeof_expr e1) (ctx |> Ctx.break_ty loc) in
      (EBreak (loc, t, e1), ctx)
  | EContinue (loc, t) ->
      (EContinue (loc, t), ctx)
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

     The error occurs because function pointers are values. In the general case
     we cannot know what a value is unless we evaluate the program. Thus, it
     would not be possible to know what a function pointer monomorphises to at
     compiletime. The following is possible however.

     fun does_work() {
         val x1 = duplicate;  # OK
         val x2 = 1;          # OK
         val x3 = x1(x2, x2); # OK
         val x4 = duplicate;  # OK
         val x5 = "1";        # OK
         val x6 = x4(x5, x5); # OK
     }
  *)
  | EItem (loc, t, xs, ts) ->
      let (ts, ctx) = infer_item_expr loc t (xs, ts) ctx in
      (EItem (loc, t, xs, ts), ctx)
  | ECallExpr (loc, t, e1, es) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let (es, ctx) = es |> mapm infer_expr ctx in
      let ctx = ctx |> unify loc (typeof_expr e1) (TFunc (ts_of_es es, t)) in
      (ECallExpr (loc, t, e1, es), ctx)
  | ECallItem (loc, t, xs, ts, es) ->
      let (es, ctx) = es |> mapm infer_expr ctx in
      let (ts, ctx) = infer_item_expr loc (TFunc (ts_of_es es, t)) (xs, ts) ctx in
      (ECallItem (loc, t, xs, ts, es), ctx)
  | ESpawn (loc, t, xs, ts, es) ->
      let (es, ctx) = es |> mapm infer_expr ctx in
      let ctx = ctx |> unify loc t (atom "unit") in
      let (ts, ctx) = infer_item_expr loc (TFunc (ts_of_es es, (atom "never"))) (xs, ts) ctx in
      (ESpawn (loc, t, xs, ts, es), ctx)
  | EVar (loc, t, x) ->
      let t1 = (ctx |> Ctx.typeof_var loc x) in
      let ctx = ctx |> unify loc t t1 in
      (EVar (loc, t, x), ctx)
  | EMatch (loc, t1, e1, arms) ->
      let (e1, ctx) = ctx |> infer_expr e1 in
      let (arms, ctx) = arms |> mapm (infer_arm (typeof_expr e1) t1 loc) ctx in
      (EMatch (loc, t1, e1, arms), ctx)

and infer_expr_field (x, e) ctx =
  let (e, ctx) = ctx |> infer_expr e in
  ((x, e), ctx)

and infer_pat_field (x, p) ctx =
  let (p, ctx) = ctx |> infer_pat p in
  ((x, p), ctx)

and infer_arm t1 t2 loc (p, b) ctx =
  let (p, ctx) = ctx |> infer_pat p in
  let (b, ctx) = ctx |> infer_block b in
  let ctx = ctx |> unify loc t1 (typeof_pat p) in
  let ctx = ctx |> unify loc t2 (typeof_block b) in
  ((p, b), ctx)

and infer_pat p ctx =
  match p with
  | PVar (loc, t, x) ->
      let ctx = ctx |> Ctx.bind_var x t in
      (PVar (loc, t, x), ctx)
  | PIgnore (loc, t) ->
      (PIgnore (loc, t), ctx)
  | POr (loc, t, p0, p1) ->
      let (p0, ctx) = ctx |> infer_pat p0 in
      let (p1, ctx) = ctx |> infer_pat p1 in
      let ctx = unify loc t (typeof_pat p0) ctx in
      let ctx = unify loc t (typeof_pat p1) ctx in
      (POr (loc, t, p0, p1), ctx)
  | PRecord (loc, t, (xps, p1)) ->
      let (xps, ctx) = xps |> mapm infer_pat_field ctx in
      let (p1, ctx) = ctx |> infer_opt infer_pat p1 in
      let xts = xts_of_xps xps in
      let (r, ctx) = match p1 with
        | None -> (TRowEmpty, ctx)
        | Some p1 ->
            let (r, ctx) = ctx |> Ctx.fresh_r in
            let ctx = ctx |> unify loc (TRecord r) (typeof_pat p1) in
            (r, ctx)
      in
      let ctx = ctx |> unify loc t (TRecord (xts |> fields_to_rows r)) in
      (PRecord (loc, t, (xps, p1)), ctx)
  | PConst (loc, t, l) ->
      let ctx = ctx |> infer_lit loc t l in
      (PConst (loc, t, l), ctx)
  | PUnwrap (loc, t, x, p1) ->
      let (p1, ctx) = ctx |> infer_pat p1 in
      let (r, ctx) = ctx |> Ctx.fresh_r in
      let ctx = ctx |> unify loc t (TEnum r) in
      let ctx = ctx |> unify loc t (TEnum ([(x, typeof_pat p1)] |> fields_to_rows r)) in
      (PUnwrap (loc, t, x, p1), ctx)

and infer_item_expr loc t (xs, ts0) ctx =
  let i = ctx.ir1 |> get_item loc xs in
  let ctx = ctx |> infer_item (xs, i) in
  let sc = ctx |> Ctx.get_scheme xs in
  match sc with
  (* If we have already inferred this item then we can just instantiate
  ** its type scheme *)
  | Ctx.SPoly sc ->
      let (ts1, s0, ctx) = ctx |> instantiate_generics sc.explicit_gs in
      let (ts2, s1, ctx) = ctx |> instantiate_generics sc.implicit_gs in
      let ctx = ctx |> unify_ts loc ts0 ts1 in
      let t1 = sc.t_fun |> instantiate (s0 @ s1) in
      let ctx = ctx |> unify loc t t1 in
      (ts1 @ ts2, ctx)
  (* If we are currently infering this item, then we must reuse its
  ** implicit types to avoid polymorphic infinite recursion. *)
  | Ctx.SMono sc ->
      let (ts1, s, ctx) = ctx |> instantiate_generics sc.explicit_gs in
      let ctx = ctx |> unify_ts loc ts0 ts1 in
      let t1 = sc.t_fun |> instantiate s in
      let ctx = ctx |> unify loc t t1 in
      (ts1, ctx)
(*
type Id = String
data Kind = Star | Kfun Kind Kind
data Type = TVar Tyvar | TCon Tycon | TAp Type Type | TGen Int
data Tyvar = Tyvar Id Kind
data Tycon = Tycon Id Kind
data Qual t = [Pred] :=> t
data Pred = IsIn Id Type
type Class = ([Id], [Inst])
type Inst = Qual Pred
data ClassEnv = ClassEnv {classes :: Id -> Maybe Class, defaults :: [Type]}
type EnvTransformer = ClassEnv -> Maybe ClassEnv
data Scheme = Forall [Kind] (Qual Type)
data Assump = Id :>: Scheme
newtype TI a = TI (Subst -> Int -> (Subst, Int, a))
type Infer e t = ClassEnv -> [Assump] -> e -> TI ([Pred], t)
data Literal = LitInt Integer | LitChar Char | LitRat Rational | LitStr String
data Pat = PVar Id | PWildcard | PAs Id Pat | PLit Literal | PNpk Id Integer | PCon Assump [Pat]
data Expr = Var Id | Lit Literal | Const Assump | Ap Expr Expr | Let BindGroup Expr
type Alt = ([Pat], Expr )
type Ambiguity = (Tyvar, [Pred])
type Expl = (Id, Scheme, [Alt])
type Impl = (Id, [Alt])
type BindGroup = ([Expl], [[Impl]])
 *)


