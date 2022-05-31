open Utils
open Ir1

module NameSet = Set.Make(struct type t = name let compare = compare end)
module PathMap = Map.Make(struct type t = path let compare = compare end)
module NameMap = Map.Make(struct type t = name let compare = compare end)

module Ctx = struct
  type t = {
    ir1: ir1;
    ir2: Ir2.ir2;
    vstack: vscope list;
    next_var_uid: Gen.t;
  }
  and vscope = {
    stmts: Ir2.stmt list;
    vsubst: (name * name) list;
  }

  let rec make ir1 = {
    ir1;
    ir2 = [];
    vstack = [];
    next_var_uid = Gen.make ();
  }

  and fresh_x ctx =
    let (i, next_var_uid) = ctx.next_var_uid |> Gen.fresh in
    let v = (sprintf "v%d" i) in
    let ctx = { ctx with next_var_uid } in
    (v, ctx)

  and get_vscope ctx =
    ctx.vstack |> hd

  and update_vscope f ctx =
    { ctx with vstack = (f (hd ctx.vstack))::(tl ctx.vstack) }

  and push_vscope ctx =
    { ctx with vstack = { stmts = []; vsubst = []; }::ctx.vstack }

  and pop_vscope ctx =
    let vscope = hd ctx.vstack in
    let vstack = tl ctx.vstack in
    (vscope.stmts |> rev, { ctx with vstack })

  and add_expr t e ctx =
    let (x, ctx) = ctx |> fresh_x in
    let s = Ir2.SVal (x, t, e) in
    let ctx = ctx |> update_vscope (fun vscope -> { vscope with stmts = s::vscope.stmts }) in
    (x, ctx)

  and add_stmts ss ctx =
    ctx |> update_vscope (fun vscope -> { vscope with stmts = (ss |> rev) @ vscope.stmts })

  and add_item xs i ctx =
    { ctx with ir2 = ((xs, i)::ctx.ir2) }

  and subst_var v0 v1 ctx =
    ctx |> update_vscope (fun vscope -> { vscope with vsubst = (v0, v1)::vscope.vsubst })

  and find_var v ctx =
    match ctx.vstack |> List.find_map (fun vscope -> vscope.vsubst |> List.assoc_opt v) with
    | Some v -> find_var v ctx (* Could be a transitive thing *)
    | None -> v

end

let rec ir1_to_ir2 ir1 =
  let ctx = Ctx.make ir1 in
  let ctx = ir1 |> foldl (fun ctx i -> lower_item i ctx) ctx in
  ctx.ir2 |> rev

and lower_block (ss, e) ctx =
  let ctx = ctx |> Ctx.push_vscope in
  let ctx = ss |> foldl (fun ctx s -> lower_stmt s ctx) ctx in
  let (v, ctx) = ctx |> lower_expr e in
  let (ss, ctx) = ctx |> Ctx.pop_vscope in
  let b = (ss, v) in
  (b, ctx)

and lower_stmt s ctx =
  match s with
  | SExpr e ->
      let (_, ctx) = ctx |> lower_expr e in
      ctx

and lower_param (x, t) ctx =
  let (t, ctx) = ctx |> lower_type t in
  ((x, t), ctx)

and lower_item (xs, i) ctx =
  match i with
  | IDef (loc, a, gs, ps, t, _bs, b) ->
      let (ps, ctx) = ps |> mapm lower_param ctx in
      let (t, ctx) = ctx |> lower_type t in
      let (b, ctx) = ctx |> lower_block b in
      ctx |> Ctx.add_item xs (Ir2.IDef (loc, a, gs, ps, t, b))
  | IExternDef (loc, d, async, gs, ts, t, _bs) ->
      let (t, ctx) = ctx |> lower_type t in
      let (ts, ctx) = ts |> mapm lower_type ctx in
      ctx |> Ctx.add_item xs (Ir2.IExternDef (loc, d, async, gs, ts, t))
  | IExternType (loc, d, gs, _bs) ->
      ctx |> Ctx.add_item xs (Ir2.IExternType (loc, d, gs))
  | IType (loc, d, gs, t, _bs) ->
      let (t, ctx) = ctx |> lower_type t in
      ctx |> Ctx.add_item xs (Ir2.IType (loc, d, gs, t))
  | IVal (loc, d, t, b) ->
      let (t, ctx) = ctx |> lower_type t in
      let (b, ctx) = ctx |> lower_block b in
      ctx |> Ctx.add_item xs (Ir2.IVal (loc, d, t, b))
  | IClass _ -> todo ()

and lower_type t ctx =
  match t with
  | TFunc (ts, t) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = ctx |> lower_type t in
      (Ir2.TFunc (ts, t), ctx)
  | TRecord t ->
      let (t, ctx) = ctx |> lower_type t in
      (Ir2.TRecord t, ctx)
  | TEnum t ->
      let (t, ctx) = ctx |> lower_type t in
      (Ir2.TEnum t, ctx)
  | TRowEmpty ->
      (Ir2.TRowEmpty, ctx)
  | TRowExtend ((x, t), r) ->
      let (t, ctx) = lower_type t ctx in
      let (r, ctx) = lower_type r ctx in
      (Ir2.TRowExtend ((x, t), r), ctx)
  | TNominal (xs, ts) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      (Ir2.TNominal (xs, ts), ctx)
  | TGeneric x -> (Ir2.TGeneric x, ctx)
  | TInverse t ->
      begin match t with
      | TInverse t -> ctx |> lower_type t
      | TNominal (["std"; "PushChan"], ts) ->
          ctx |> lower_type (TNominal (["std"; "PullChan"], ts))
      | _ -> unreachable ()
      end
  | TVar x -> panic ("ICE: Found type variable in inferred code: '" ^ x)

and lower_expr e ctx =
  let (t, ctx) = ctx |> lower_type (Ir1.typeof_expr e) in
  match e with
  | EAccess (loc, _, e1, x) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      ctx |> Ctx.add_expr t (Ir2.EAccess (loc, v1, x))
  | EUpdate (loc, _, e1, x, e2) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      let (v2, ctx) = ctx |> lower_expr e2 in
      ctx |> Ctx.add_expr t (Ir2.EUpdate (loc, v1, x, v2))
  | ECast (loc, _, e1, t2) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      let (t2, ctx) = ctx |> lower_type t2 in
      ctx |> Ctx.add_expr t (Ir2.ECast (loc, v1, t2))
  | EOn _ ->
      todo ()
  | ELit (loc, _, l) ->
      ctx |> Ctx.add_expr t (Ir2.ELit (loc, l))
  | ELoop (loc, _, b) ->
      let (b, ctx) = ctx |> lower_block b in
      ctx |> Ctx.add_expr t (Ir2.ELoop (loc, b))
  | ERecord (loc, _, (xes, e)) ->
      let (xes, ctx) = xes |> mapm lower_expr_field ctx in
      let (e, ctx) = ctx |> lower_expr_opt e in
      ctx |> Ctx.add_expr t (Ir2.ERecord (loc, (xes, e)))
  | EReturn (loc, _, e1) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      ctx |> Ctx.add_expr t (Ir2.EReturn (loc, v1))
  | EBreak (loc, _, e1) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      ctx |> Ctx.add_expr t (Ir2.EBreak (loc, v1))
  | EContinue (loc, _) ->
      ctx |> Ctx.add_expr t (Ir2.EContinue (loc))
  | EEnwrap (loc, _, xs, e1) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      ctx |> Ctx.add_expr t (Ir2.EEnwrap (loc, xs, v1))
  | EItem (loc, _, xs, ts) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      ctx |> Ctx.add_expr t (Ir2.EItem (loc, xs, ts))
  | ECallExpr (loc, _, e1, es) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      let (vs, ctx) = es |> mapm lower_expr ctx in
      ctx |> Ctx.add_expr t (Ir2.ECallExpr (loc, v1, vs))
  | ECallItem (loc, _, xs, ts, es) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (vs, ctx) = es |> mapm lower_expr ctx in
      ctx |> Ctx.add_expr t (Ir2.ECallItem (loc, xs, ts, vs))
  | EVar (_, _, x) ->
      let x = ctx |> Ctx.find_var x in
      (x, ctx)
  | EMatch (_, _, e1, arms) ->
      let (v1, ctx) = ctx |> lower_expr e1 in
      (* let (arms, ctx) = arms |> mapm lower_arm ctx in *)
      let clauses = arms_to_clauses arms v1 in
      let ((ss, v), ctx) = ctx |> lower_pat_clauses t clauses in
      let ctx = ctx |> Ctx.add_stmts ss in
      (v, ctx)
  | ESpawn (loc, _, xs, ts, es) ->
      let (vs, ctx) = es |> mapm lower_expr ctx in
      let (ts, ctx) = ts |> mapm lower_type ctx in
      ctx |> Ctx.add_expr t (Ir2.ESpawn (loc, xs, ts, vs))

and lower_expr_field (x, e) ctx =
  let (e, ctx) = ctx |> lower_expr e in
  ((x, e), ctx)

and lower_expr_opt e ctx =
  match e with
  | Some e ->
      let (e, ctx) = ctx |> lower_expr e in
      (Some e, ctx)
  | None -> (None, ctx)

(* Convert match arms to clauses. A clause has the following form: `(eqs, substs, expr)` where:
** - `eqs` are a set of equations of the form `(v, p)` where
**   - `v` is a variable
**   - `p` is a pattern match on the variable `v`
** - `substs` are a set of substitutions of the form `(v0, v1)` where
**   - `v0` is substituted for `v1` inside `expr`
** - `expr` is an expression which is evaluated if the clause succeeds
*)
and arms_to_clauses arms v =
  arms |> List.map (fun (p, b) -> ([(v, p)], [], b))

(* For all variables in the head clauses' equations,
   return the variable that occurs in the most equations of all clauses *)
and branch_heuristic (eqs, _, _) cs =
  eqs |> map (fun (x, _) -> x)
      |> max_by (fun x ->
        cs |> List.filter (fun (eqs, _, _) ->
          eqs |> List.exists (fun (y, _) -> x = y)))

and simplify_pat_field loc v ((eqs, substs, b), (cs, ctx)) (x, p) =
  let (t, ctx) = ctx |> lower_type (typeof_pat p) in
  let (v, ctx) = ctx |> Ctx.add_expr t (Ir2.EAccess (loc, v, x)) in
  simplify_pat_eq ((eqs, substs, b), (cs, ctx)) (v, p)

(* Simplifies an equation v = p such that it only contains refutable patterns. *)
and simplify_pat_eq ((eqs, substs, b), (cs, ctx)) (v, p) =
  match p with
  | Ir1.PIgnore _ ->
      ((eqs, substs, b), (cs, ctx))
  | Ir1.PRecord (loc, _, (xps, _p)) ->
      xps |> foldl (simplify_pat_field loc v) ((eqs, substs, b), (cs, ctx))
  | Ir1.PVar (_, _, v0) ->
      ((eqs, (v0, v)::substs, b), (cs, ctx))
  | Ir1.POr (_, _, p0, p1) ->
      let (v0, ctx) = ctx |> Ctx.fresh_x in
      let (v1, ctx) = ctx |> Ctx.fresh_x in
      let eqs0 = (v0, p0)::eqs in
      let eqs1 = (v1, p1)::eqs in
      let (c0, (cs, ctx)) = simplify_clause (eqs0, substs, b) (cs, ctx) in
      let (c1, (cs, ctx)) = simplify_clause (eqs1, substs, b) (cs, ctx) in
      ((eqs, substs, b), (c0::c1::cs, ctx))
  | Ir1.PConst _ | Ir1.PUnwrap _ ->
      (((v, p)::eqs, substs, b), (cs, ctx))

and simplify_clause (eqs, substs, b) (cs, ctx) =
  eqs |> foldl simplify_pat_eq (([], substs, b), (cs, ctx))

and lower_pat_clauses match_t clauses ctx =
  let (clauses, (clauses', ctx)) = clauses |> mapm simplify_clause ([], ctx) in
  let clauses = clauses @ clauses' in
  let (eqs, substs, b) as head_clause = clauses |> List.hd in
  if eqs = [] then (* This pattern equation is now solved *)
    let ctx = substs |> foldl (fun ctx (v0, v1) -> ctx |> Ctx.subst_var v0 v1) ctx in
    lower_block b ctx (* NOTE: Duplication leads to duplicated lowerings, we might need sharing. *)
  else
    (* Branch on the variable in the head clause which occurs in the most equations *)
    let branch_v = branch_heuristic head_clause clauses in
    match eqs |> List.assoc branch_v with
    | Ir1.PUnwrap (loc, _, variant_x, p) ->
        let (unwrap_t, ctx) = ctx |> lower_type (typeof_pat p) in
        branch_pat_unwrap_clause match_t loc branch_v unwrap_t variant_x clauses ctx
    | Ir1.PConst (loc, t, l) ->
        let (const_t, ctx) = ctx |> lower_type t in
        let const_l = l in
        branch_pat_const_clause match_t loc branch_v const_t const_l clauses ctx
    (* Irrefutable top-level patterns are eliminated earlier through simplify_clauses *)
    | Ir1.PVar _ | Ir1.PIgnore _ | Ir1.PRecord _ | Ir1.POr _ -> unreachable ()

and lower_branch_unwrap match_t clauses unwrap_s ctx =
  let ctx = ctx |> Ctx.push_vscope in
  let ((ss0, v), ctx) = lower_pat_clauses match_t clauses ctx in
  let (ss1, ctx) = ctx |> Ctx.pop_vscope in
  ((unwrap_s::(ss1 @ ss0), v), ctx)

and branch_pat_unwrap_clause match_t loc branch_v unwrap_t variant_x clauses ctx =
  (* Create fresh variable for the inner pattern *)
  let (unwrap_v, ctx) = ctx |> Ctx.fresh_x in
  let unwrap_e = Ir2.EUnwrap (loc, variant_x, branch_v) in
  let unwrap_s = Ir2.SVal (unwrap_v, unwrap_t, unwrap_e) in

  let (then_clauses, else_clauses, ctx) = ctx |> split_pat_unwrap_clauses clauses branch_v unwrap_v variant_x in

  match (then_clauses, else_clauses) with
  | (then_clauses, []) ->
      lower_branch_unwrap match_t then_clauses unwrap_s ctx
  | ([], else_clauses) ->
      lower_pat_clauses match_t else_clauses ctx
  | (then_clauses, else_clauses) ->
      let (v_check, ctx) = ctx |> Ctx.add_expr (Ir2.atom "bool") (Ir2.ECheck (loc, variant_x, branch_v)) in
      let (then_b, ctx) = lower_branch_unwrap match_t then_clauses unwrap_s ctx in
      let (else_b, ctx) = lower_pat_clauses match_t else_clauses ctx in
      let (v, ctx) = ctx |> Ctx.add_expr match_t (Ir2.EIf (loc, v_check, then_b, else_b)) in
      (([], v), ctx)

and branch_pat_const_clause match_t loc branch_v const_t const_l clauses ctx =
  let (then_clauses, else_clauses, ctx) = ctx |> split_pat_const_clauses clauses branch_v const_l in
  let (v, ctx) = ctx |> Ctx.add_expr (Ir2.atom "bool") (Ir2.ECallItem (loc, ["std"; "eq"], [const_t], [branch_v])) in

  match (then_clauses, else_clauses) with
  | (then_clauses, []) ->
      lower_pat_clauses match_t then_clauses ctx
  | ([], else_clauses) ->
      lower_pat_clauses match_t else_clauses ctx
  | (then_clauses, else_clauses) ->
      let (then_b, ctx) = lower_pat_clauses match_t then_clauses ctx in
      let (else_b, ctx) = lower_pat_clauses match_t else_clauses ctx in
      let (v, ctx) = ctx |> Ctx.add_expr match_t (Ir2.EIf (loc, v, then_b, else_b)) in
      (([], v), ctx)

and split_pat_unwrap_clauses clauses branch_v unwrap_v variant_x ctx =
  let split_pat_clause (then_clauses, else_clauses, ctx) ((eqs, substs, e) as clause) =  
    match eqs |> List.assoc_opt branch_v with
    | Some Ir1.PUnwrap (_, _, x, p) ->
        if variant_x = x then
          (* Push clauses with pattern matches on the branching variable to the then-branch *)
          let eqs = (unwrap_v, p)::(eqs |> List.remove_assoc branch_v) in
          let clause = (eqs, substs, e) in
          (clause::then_clauses, else_clauses, ctx)
        else 
          (* Push clauses without pattern matches on the branching variable to the else-branch *)
          (then_clauses, clause::else_clauses, ctx)
    | None ->
        (* Clauses which do not match on the branching variable are pushed to both branches *)
        (clause::then_clauses, clause::else_clauses, ctx)
    | _ -> unreachable ()
  in
  let (then_clauses, else_clauses, ctx) = clauses |> foldl split_pat_clause ([], [], ctx) in
  (then_clauses |> rev, else_clauses |> rev, ctx)

and split_pat_const_clauses clauses branch_v const_l ctx =
  let split_pat_clause (then_clauses, else_clauses, ctx) ((eqs, substs, e) as clause) =  
    match eqs |> List.assoc_opt branch_v with
    | Some Ir1.PConst (_, _, l) ->
        if const_l = l then
          let eqs = eqs |> List.remove_assoc branch_v in
          let clause = (eqs, substs, e) in
          (clause::then_clauses, else_clauses, ctx)
        else 
          (then_clauses, clause::else_clauses, ctx)
    | None -> (clause::then_clauses, clause::else_clauses, ctx)
    | _ -> unreachable ()
  in
  let (then_clauses, else_clauses, ctx) = clauses |> foldl split_pat_clause ([], [], ctx) in
  (then_clauses |> rev, else_clauses |> rev, ctx)
