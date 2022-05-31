open Utils
open Ir1

module Ctx = struct
  type t = {
    graph: Graph.t;          (* Graph of top-level item declarations *)
    next_def_uid: Gen.t;     (* Function uid counter *)
    next_type_uid: Gen.t;    (* Type uid counter *)
    next_generic_uid: Gen.t; (* Generic uid counter *)
    next_expr_uid: Gen.t;    (* Expression uid counter *)
    vstack: vscope list;     (* Stack of scopes for value parameters *)
    gstack: gscope list;     (* Stack of scopes for type parameters *)
    path: path;              (* Current path *)
    ir1: definition list;    (* Final output of the lowering *)
    astack: ascope list;     (* Stack of anonymous variables *)
  }
  and mut = MVar | MVal
  and vscope = { vsubsts: (name * (name * mut)) list; stmts: stmt list }
  and gscope = { gsubsts: (name * name) list }
  and ascope = { asubsts: name list }
  and definition = path * item

  let rec make graph = {
    graph;
    next_def_uid = Gen.make ();
    next_type_uid = Gen.make ();
    next_generic_uid = Gen.make ();
    next_expr_uid = Gen.make ();
    vstack = [];
    gstack = [];
    astack = [];
    path = [];
    ir1 = [];
  }

  and add_item xs i ctx = { ctx with ir1 = (xs, i)::ctx.ir1 }

  and new_expr f ctx =
    let (t, ctx) = ctx |> fresh_t in
    (f t, ctx)

  and new_pat f ctx =
    let (t, ctx) = ctx |> fresh_t in
    (f t, ctx)

  and fresh_ts n ctx =
    repeat fresh_t n ctx

  and fresh_t ctx =
    let (n, next_type_uid) = ctx.next_type_uid |> Gen.fresh in
    let ctx = { ctx with next_type_uid } in
    let x = Printf.sprintf "%d" n in
    let t = TVar x in
    (t, ctx)

  and fresh_xs n ctx =
    repeat fresh_x n ctx

  and fresh_x ctx =
    let (n, next_expr_uid) = ctx.next_expr_uid |> Gen.fresh in
    let x = Printf.sprintf "x%d" n in
    let ctx = { ctx with next_expr_uid; } in
    (x, ctx)

  and fresh_ps n ctx =
    let (xs, ctx) = fresh_xs n ctx in
    let (ts, ctx) = fresh_ts n ctx in
    (zip xs ts, ctx)

  and fresh_f ctx =
    let (n, next_def_uid) = ctx.next_def_uid |> Gen.fresh in
    let x = Printf.sprintf "f%d" n in
    let ctx = { ctx with next_def_uid; } in
    ([x], ctx)

  and fresh_g ctx =
    let (n, next_generic_uid) = ctx.next_generic_uid |> Gen.fresh in
    let g = Printf.sprintf "T%d" n in
    let ctx = { ctx with next_generic_uid; } in
    (g, ctx)

  and push_vscope ctx =
    { ctx with vstack = {vsubsts=[];stmts=[]}::ctx.vstack }

  and pop_vscope ctx =
    ((hd ctx.vstack).stmts |> rev, { ctx with vstack = tl ctx.vstack })

  and pop_vscope_inlined ctx =
    let (es, ctx) = pop_vscope ctx in
    ctx |> add_stmts es

  and pop_vscope_to_block e ctx =
    let (ss, ctx) = pop_vscope ctx in
    ((ss, e), ctx)

  and enter_vscope f ctx =
    let ctx = ctx |> push_vscope in
    let (e, ctx) = f ctx in
    let (es, ctx) = ctx |> pop_vscope in
    ((e, es), ctx)

  and add_stmts ss ctx =
    match ctx.vstack with
    | h::t -> { ctx with vstack = { h with stmts = ss @ h.stmts}::t }
    | [] -> unreachable ()

  and add_stmt_expr e ctx =
    match ctx.vstack with
    | h::t -> { ctx with vstack = { h with stmts = (SExpr e)::h.stmts}::t }
    | [] -> unreachable ()

  and push_gscope ctx =
    { ctx with gstack = {gsubsts=[]}::ctx.gstack }

  and pop_gscope (ctx:t) = { ctx with gstack = tl ctx.gstack }

  and push_ascope (ctx:t) = { ctx with astack = { asubsts=[] }::ctx.astack }

  and pop_ascope (ctx:t) =
    let vs = (hd ctx.astack).asubsts in
    let ctx = { ctx with astack = tl ctx.astack } in
    (vs, ctx)

  and add_anon (ctx:t) =
    match ctx.astack with
    | h::t ->
        let (x, ctx) = ctx |> fresh_x in
        let ctx = { ctx with astack = { asubsts=x::h.asubsts }::t } in
        (x, ctx)
    | [] -> unreachable ()

  (* Returns a name path *)
  and item_path x ctx = x::ctx.path |> rev

  and bind_gname g (ctx:t) =
    match ctx.gstack with
    | h::t ->
        let (g', ctx) = ctx |> fresh_g in
        let gstack = { gsubsts = (g, g')::h.gsubsts }::t in
        let ctx = { ctx with gstack } in
        (g', ctx)
    | [] -> unreachable ()

  and bind_vname v m (ctx:t) =
    match ctx.vstack with
    | h::t ->
        let (v', ctx) = ctx |> fresh_x in
        let vstack = {h with vsubsts = (v, (v', m))::h.vsubsts}::t in
        let ctx = { ctx with vstack } in
        (v', ctx)
    | [] -> unreachable ()

  (* Finds a value variable. Note that this implementation allows shadowing
     since we look at the most recently bound variables first. *)
  and find_vname_opt v (ctx:t) =
    ctx.vstack |> List.find_map (fun vscope -> vscope.vsubsts |> List.assoc_opt v)

  and find_vname loc v ctx =
    match ctx |> find_vname_opt v with
    | Some v -> v
    | None ->
        if !Args.verbose then begin
          Printf.printf "Currently bound variables: ";
          debug_vstack ctx
        end;
        raise (Error.NamingError (loc, "Undefined variable `" ^ v ^ "`"))

  and debug_vstack ctx =
    let rec debug_vstack vstack =
      match vstack with
      | h::t ->
          h.vsubsts |> List.iter (function
            | (x0, (x1, MVal)) -> Printf.printf "%s -> var %s " x0 x1
            | (x0, (x1, MVar)) -> Printf.printf "%s -> val %s " x0 x1
          );
          debug_vstack t
      | [] -> ()
    in
    debug_vstack ctx.vstack

  (* Finds a generic variable *)
  and find_gname_opt g (ctx:t) =
    ctx.gstack |> List.find_map (fun gscope -> gscope.gsubsts |> List.assoc_opt g)

  and find_gname loc g ctx =
    match ctx |> find_gname_opt g with
    | Some g -> g
    | None -> raise (Error.NamingError (loc, "Undefined generic `" ^ g ^ "`"))

  and push_namespace x ctx =
    let ctx = { ctx with path = x::ctx.path } in
    (ctx.path |> rev, ctx)

  and pop_namespace ctx = { ctx with path = ctx.path |> List.tl }

  and resolve_path_opt xs ctx =
    match xs with
    | Ast.PAbs xs -> ctx.graph |> Graph.resolve_path xs
    | Ast.PRel xs -> ctx.graph |> Graph.resolve_path ((rev ctx.path) @ xs)

  and resolve_path loc xs ctx =
    match resolve_path_opt xs ctx with
    | Some ((xs, d)) -> (xs, d)
    | None ->
        if !Args.verbose then begin
          Printf.printf "%s" (String.concat "::" ctx.path);
        end;
        raise (Error.NamingError (loc, "Path is not bound to anything: " ^ (Print.Ast.path_to_str xs)))

  (* Returns set of currently visible variables *)
  and visible ctx =
     ctx.vstack |> foldl (fun acc vscope -> vscope.vsubsts |> foldl (fun acc v -> v::acc) acc) []

end

let rec ast_to_ir1 graph ast =
  let ctx = Ctx.make graph in
  let ctx = ast |> foldl (fun ctx i -> lower_item i ctx) ctx in
  let ir1 = ctx.ir1 |> rev in
  ir1

and lower_item i ctx =
  match i with
  | Ast.IVal (loc, d, x, t, e) ->
      let xs = ctx |> Ctx.item_path x in
      let (t, ctx) = lower_type_or_fresh t ctx in
      let ctx = ctx |> Ctx.push_vscope in
      let (e, ctx) = lower_expr e ctx in
      let (b, ctx) = ctx |> Ctx.pop_vscope_to_block e in
      ctx |> Ctx.add_item xs (IVal (loc, d, t, b))
  | Ast.IExternDef (loc, d, async, x, gs, ts, t, bs) ->
      let x = Ast.def_name x in
      let xs = ctx |> Ctx.item_path x in
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = lower_type_or_unit t ctx in
      let (bs, ctx) = bs |> mapm (lower_bound loc) ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      let ctx = ctx |> Ctx.add_item xs (IExternDef (loc, d, async, gs, ts, t, bs)) in
      if async then
        ctx
      else
        let (vs, ctx) = ctx |> Ctx.fresh_xs (List.length ts) in
        let ps = zip vs ts in
        ctx |> add_indirect_def loc x gs ps t bs
  | Ast.IExternType (loc, d, x, gs, bs) ->
      let xs = ctx |> Ctx.item_path x in
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (bs, ctx) = bs |> mapm (lower_bound loc) ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (IExternType (loc, d, gs, bs))
  | Ast.IDef (loc, d, _async, x, gs, pts, t, bs, b) ->
      let x = Ast.def_name x in
      let xs = ctx |> Ctx.item_path x in
      let ctx = ctx |> Ctx.push_gscope in
      let ctx = ctx |> Ctx.push_vscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (ps, ctx) = pts |> map fst |> mapm lower_pat ctx in
      let (ts, ctx) = pts |> map snd |> mapm lower_type_or_fresh ctx in
      let (vs, ctx) = pts |> mapm (fun _ ctx -> Ctx.fresh_x ctx) ctx in
      let (t, ctx) = ctx |> lower_type_or_fresh t in
      let (bs, ctx) = bs |> mapm (lower_bound loc) ctx in
      let ((es0, e), ctx) = ctx |> lower_block b in
      let (es1, ctx) = ctx |> Ctx.pop_vscope in
      let b = (es1 @ es0, e) in
      let ctx = ctx |> Ctx.pop_gscope in
      if vs <> [] then
        let (e, ctx) = ctx |> vars_to_expr_record loc vs in
        let (p, ctx) = ctx |> patterns_to_record loc ps in
        let (e, ctx) = ctx |> Ctx.new_expr (fun t -> EMatch (loc, t, e, [(p, b)])) in
        let vts = zip vs ts in
        ctx |> Ctx.add_item xs (IDef (loc, d, gs, vts, t, bs, ([], e)))
            |> add_indirect_def loc x gs vts t bs
      else
        ctx |> Ctx.add_item xs (IDef (loc, d, gs, [], t, bs, b))
            |> add_indirect_def loc x gs [] t bs
  | Ast.ITask (loc, d, x, gs, pts, xts, bs, b) ->
      let x = Ast.def_name x in
      let xs = ctx |> Ctx.item_path x in
      let xs_impl = ctx |> Ctx.item_path (Printf.sprintf "%s_impl" x) in
      ctx |> add_task_impl_def loc d xs_impl gs pts xts bs b
          |> add_task_def loc d xs xs_impl gs pts xts bs
  | Ast.IType (loc, d, x, gs, t, bs) ->
      let xs = ctx |> Ctx.item_path x in
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (t, ctx) = lower_type t ctx in
      let (bs, ctx) = bs |> mapm (lower_bound loc) ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (IType (loc, d, gs, t, bs))
  | Ast.IMod (_, _, x, is) ->
      let (_, ctx) = ctx |> Ctx.push_namespace x in
      let ctx = is |> foldl (fun ctx i -> lower_item i ctx) ctx in
      let ctx = ctx |> Ctx.pop_namespace in
      ctx
  | Ast.IUse _ -> ctx
  | Ast.IClass (_loc, _d, _x, _gs, _bs, _decls) -> todo ()
  | Ast.IInstance (_loc, _d, _gs, _xs, _ts, _bs, _defs) -> todo ()

and lower_bound loc (xs, ts) ctx =
  let (ts, ctx) = ts |> mapm lower_type ctx in
  begin match ctx |> Ctx.resolve_path loc xs with
  | (xs, Graph.NItem _) -> ((xs, ts), ctx)
  |  _ -> raise (Error.NamingError (loc, "Expected class, found something else."))
  end

and add_task_impl_def loc d xs gs pts0 xts1 bs b ctx =
  let ctx = ctx |> Ctx.push_gscope in
  let ctx = ctx |> Ctx.push_vscope in
  let (gs, ctx) = gs |> mapm lower_generic ctx in

  let (ps0, ctx) = pts0 |> map fst |> mapm lower_pat ctx in
  let (ts0, ctx) = pts0 |> map snd |> mapm lower_type_or_fresh ctx in
  let (vs0, ctx) = pts0 |> mapm (fun _ ctx -> Ctx.fresh_x ctx) ctx in

  let (xts1, ctx) = xts1 |> mapm lower_sink ctx in
  let (bs, ctx) = bs |> mapm (lower_bound loc) ctx in

  let ((ss0, v), ctx) = lower_block b ctx in
  let (ss1, ctx) = ctx |> Ctx.pop_vscope in
  let b = (ss0 @ ss1, v) in
  let ctx = ctx |> Ctx.pop_gscope in

  if vs0 <> [] then
    let (e, ctx) = ctx |> vars_to_expr_record loc vs0 in
    let (p, ctx) = ctx |> patterns_to_record loc ps0 in
    let (e, ctx) = ctx |> Ctx.new_expr (fun t -> EMatch (loc, t, e, [(p, b)])) in
    let xts = (zip vs0 ts0) @ xts1 in
    ctx |> Ctx.add_item xs (IDef (loc, d, gs, xts, atom "never", bs, ([], e)))
  else
    ctx |> Ctx.add_item xs (IDef (loc, d, gs, xts1, atom "never", bs, b))

and lower_channels loc xts ctx =
  let (vs_push, vs_pull, es, ps, ctx) = xts |> foldl (fun acc xt -> lower_channel loc xt acc) ([], [], [], [], ctx) in
  (vs_push |> rev, vs_pull |> rev, es |> rev, ps |> rev, ctx)

and lower_channel loc (x_push, t_push) (vs_push, vs_pull, es, ps, ctx) =
  let (x_push, ctx) = ctx |> Ctx.bind_vname x_push MVal in
  let (x_pull, ctx) = ctx |> Ctx.fresh_x in
  let (t_push, ctx) = ctx |> lower_type_or_fresh t_push in
  let t_pull = TInverse t_push in
  let p_push = PVar (loc, t_push, x_push) in
  let p_pull = PVar (loc, t_pull, x_pull) in
  let (p, ctx) = ctx |> patterns_to_record loc [p_push; p_pull] in
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let (e, ctx) = ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "chan"], [t], [])) in
  (x_push::vs_push, x_pull::vs_pull, e::es, p::ps, ctx)

and add_task_def loc d xs xs_impl gs pts0 xts1 bs ctx =
  let ctx = ctx |> Ctx.push_gscope in
  let ctx = ctx |> Ctx.push_vscope in
  let (gs, ctx) = gs |> mapm lower_generic ctx in

  let (ts0, ctx) = pts0 |> map snd |> mapm lower_type_or_fresh ctx in
  let (vs0, ctx) = pts0 |> mapm (fun _ ctx -> Ctx.fresh_x ctx) ctx in
  let (es0, ctx) = ctx |> vars_to_exprs loc vs0 in
  let vts = zip vs0 ts0 in

  let (vs1_push, vs1_pull, es1, ps1, ctx) = ctx |> lower_channels loc xts1 in
  let (e1_chan, ctx) = ctx |> exprs_to_record loc es1 in
  let (p1_chan, ctx) = ctx |> patterns_to_record loc ps1 in

  let (es1_push, ctx) = ctx |> vars_to_exprs loc vs1_push in

  let ts = gs |> map (fun g -> TGeneric g) in
  let (e_spawn, ctx) = ctx |> Ctx.new_expr (fun t -> ESpawn (loc, t, xs_impl, ts, es0 @ es1_push)) in
  let ss = [SExpr e_spawn] in

  let (e_pull, ctx) = match (vs1_pull) with
  | ([]) ->
      ctx |> Ctx.new_expr (fun t -> ELit (loc, t, LUnit loc))
  | ([v]) ->
      ctx |> Ctx.new_expr (fun t1 -> EVar (loc, t1, v))
  | (vs) ->
      ctx |> vars_to_expr_record loc vs
  in
  let (e, ctx) = ctx |> Ctx.new_expr (fun t1 -> EMatch (loc, t1, e1_chan, [(p1_chan, (ss, e_pull))])) in

  let (ss, ctx) = ctx |> Ctx.pop_vscope in
  let b = (ss, e) in
  let ctx = ctx |> Ctx.pop_gscope in
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let (bs, ctx) = bs |> mapm (lower_bound loc) ctx in

  ctx |> Ctx.add_item xs (IDef (loc, d, gs, vts, t, bs, b))

and vars_to_exprs loc ps ctx =
  ps |> mapm (fun x ctx -> ctx |> Ctx.new_expr (fun t -> EVar (loc, t, x))) ctx

and vars_to_patterns loc ps ctx =
  ps |> mapm (fun x ctx -> ctx |> Ctx.new_expr (fun t -> PVar (loc, t, x))) ctx

and var_to_expr loc v ctx = ctx |> Ctx.new_expr (fun t -> EVar (loc, t, v))

and var_to_generic g = TGeneric g

and vars_to_expr_record loc xs ctx =
  let (es, ctx) = ctx |> vars_to_exprs loc xs in
  ctx |> exprs_to_record loc es

and exprs_to_record loc es ctx =
  let xes = indexes_to_fields es in
  ctx |> Ctx.new_expr (fun t -> ERecord (loc, t, (xes, None)))

and types_to_record ts =
  let fs = indexes_to_fields ts in
  (TRecord (fs |> fields_to_rows TRowEmpty))

and patterns_to_record loc ps ctx =
  let xps = indexes_to_fields ps in
  ctx |> Ctx.new_pat (fun t -> PRecord (loc, t, (xps, None)))

and indirect_name x = Printf.sprintf "%s_indirect" x

and add_indirect_def loc x gs vts t bs ctx =
  let xs_direct = ctx |> Ctx.item_path x in
  let xs = ctx |> Ctx.item_path (indirect_name x) in
  let (es, ctx) = vts |> map fst |> mapm (var_to_expr loc) ctx in
  let vts = vts @ [("_", TRecord TRowEmpty)] in
  let ts = gs |> map (function g -> TGeneric g) in
  let (t1, ctx) = ctx |> Ctx.fresh_t in
  let e = ECallItem (loc, t1, xs_direct, ts, es) in
  let b = ([], e) in
  ctx |> Ctx.add_item xs (IDef (loc, [], gs, vts, t, bs, b))

and lower_generic x ctx =
  let (x, ctx) = ctx |> Ctx.bind_gname x in
  (x, ctx)

and lower_sink (x, t) ctx =
  let (x, ctx) = ctx |> Ctx.bind_vname x MVal in
  let (t, ctx) = lower_type_or_fresh t ctx in
  ((x, t), ctx)

and lower_param (p, t) ctx =
  let (p, ctx) = lower_pat p ctx in
  let (t, ctx) = lower_type_or_fresh t ctx in
  ((p, t), ctx)

(* Arg expressions can contain underscores. These are captured in `ascope`. *)
and lower_expr_arg e ctx =
  let loc = Ast.expr_loc e in
  let ctx = ctx |> Ctx.push_ascope in
  let ctx = ctx |> Ctx.push_vscope in
  let (e, ctx) = lower_expr e ctx in
  let (es, ctx) = ctx |> Ctx.pop_vscope in
  let ctx = ctx |> Ctx.add_stmts es in
  let (vs, ctx) = ctx |> Ctx.pop_ascope in
  match vs with
  | [] ->
      (e, ctx)
  | vs ->
      let (vts, ctx) = vs |> mapm (fun v ctx ->
          let (t, ctx) = ctx |> Ctx.fresh_t in
          ((v, t), ctx)
      ) ctx in
      let (xs, ctx) = ctx |> Ctx.fresh_f in
      let (t, ctx) = ctx |> Ctx.fresh_t in
      let (x_env, ctx) = ctx |> Ctx.fresh_x in
      let (t_env, ctx) = ctx |> Ctx.fresh_t in
      let xt_env = (x_env, t_env) in
      let ctx = ctx |> Ctx.add_item xs (IDef (loc, [], [], vts @ [xt_env], t, [], ([], e))) in
      let (ef, ctx) = ctx |> Ctx.new_expr (fun t -> EItem (loc, t, xs, [])) in
      let (er, ctx) = ctx |> empty_expr_env loc in
      let fs = [("f", ef); ("r", er)] in
      ctx |> Ctx.new_expr (fun t -> ERecord (loc, t, (fs, None)))

and lower_indirect_call loc e es ctx =
  let (e, ctx) = lower_expr e ctx in
  let (es, ctx) = es |> mapm lower_expr_arg ctx in
  let (ef, ctx) = ctx |> Ctx.new_expr (fun t -> EAccess (loc, t, e, "f")) in
  let (er, ctx) = ctx |> Ctx.new_expr (fun t -> EAccess (loc, t, e, "r")) in
  ctx |> Ctx.new_expr (fun t -> ECallExpr (loc, t, ef, es @ [er]))

and lower_direct_call is_operator loc xs ts es ctx =
  let (es, ctx) =
    if is_operator then
      es |> mapm lower_expr ctx
    else
      es |> mapm lower_expr_arg ctx
  in
  ctx |> Ctx.new_expr (fun t -> ECallItem (loc, t, xs, ts, es))

and lower_async_call loc xs ts es ctx =
  let (es, ctx) = es |> mapm lower_expr_arg ctx in
  ctx |> Ctx.new_expr (fun t -> ECallItem (loc, t, xs, ts, es))

and lower_type_args xs ts gs ctx =
  let n = gs |> List.length in
  match List.length ts with
  | m when m = n -> ts |> mapm lower_type ctx
  | m when m = 0 -> ctx |> Ctx.fresh_ts n
  | m -> panic (Printf.sprintf "Path `%s` has wrong number of type arguments, expected %d but found %d" (Print.path_to_str xs) n m)

and lower_expr_variant loc es ctx =
  let (es, ctx) = es |> mapm lower_expr ctx in
  let fs = es |> indexes_to_fields in
  ctx |> Ctx.new_expr (fun t -> ERecord (loc, t, (fs, None)))

and lower_call is_operator loc e es ctx =
  match e with
  | Ast.EPath (_, xs, ts) ->
      let resolve_call_path xs ctx =
        begin match ctx |> Ctx.resolve_path loc xs with
        | (xs, Graph.NItem IExternDef (_, _, async, _, gs, _, _, _)) when async = true ->
            let (ts, ctx) = lower_type_args xs ts gs ctx in
            lower_async_call loc xs ts es ctx
        | (xs, Graph.NItem IDef (_, _, _, _, gs, _, _, _, _))
        | (xs, Graph.NItem IExternDef (_, _, _, _, gs, _, _, _))
        | (xs, Graph.NItem ITask (_, _, _, gs, _, _, _, _)) ->
            let (ts, ctx) = lower_type_args xs ts gs ctx in
            lower_direct_call is_operator loc xs ts es ctx
        | _ ->
            unreachable ()
        end
      in
      begin match xs with
      | Ast.PRel [x] when ts = [] ->
          begin match ctx |> Ctx.find_vname_opt x with
          | Some (_, Ctx.MVal) -> lower_indirect_call loc e es ctx
          | Some (v, Ctx.MVar) ->
              let (e, ctx) = ctx |> Ctx.new_expr (fun t -> EVar (loc, t, v)) in
              get_cell_expr loc e ctx
          | None -> ctx |> resolve_call_path xs
          end
      | _ -> ctx |> resolve_call_path xs
      end
  | _ -> lower_indirect_call loc e es ctx

and lower_expr_item_func loc xs ts gs ctx =
  let (ts, ctx) = lower_type_args xs ts gs ctx in
  let (ef, ctx) = ctx |> Ctx.new_expr (fun t -> EItem (loc, t, xs, ts)) in
  let (er, ctx) = ctx |> empty_expr_env loc in
  ctx |> Ctx.new_expr (fun t -> ERecord (loc, t, ([("f", ef); ("r", er)], None)))

and lower_expr_item_path loc xs ts ctx =
  let (xs, decl) = ctx |> Ctx.resolve_path loc xs in
  match decl with
  | Graph.NItem Ast.IExternDef (_, _, _, _, gs, _, _, _)
  | Graph.NItem Ast.ITask (_, _, _, gs, _, _, _, _)
  | Graph.NItem Ast.IDef (_, _, _, _, gs, _, _, _, _) ->
      ctx |> lower_expr_item_func loc xs ts gs
  | Graph.NItem Ast.IVal _ ->
      ctx |> Ctx.new_expr (fun t -> EItem (loc, t, xs, []))
  | Graph.NMethodDecl _
  | Graph.NItem _ ->
      raise (Error.NamingError (loc, "Found non-expr where expr was expected"))

(* Resolves a path expression *)
and lower_expr_path loc xs ts ctx =
  match xs with
  | Ast.PRel [x] when ts = [] ->
      begin match ctx |> Ctx.find_vname_opt x with
      | Some (v, Ctx.MVal) ->
          ctx |> Ctx.new_expr (fun t -> EVar (loc, t, v))
      | Some (v, Ctx.MVar) ->
          let (e, ctx) = ctx |> Ctx.new_expr (fun t -> EVar (loc, t, v)) in
          get_cell_expr loc e ctx
      | None -> ctx |> lower_expr_item_path loc xs ts
      end
  | _ -> ctx |> lower_expr_item_path loc xs ts

and lower_type_item_path loc xs ts ctx =
  let (xs, decl) = ctx |> Ctx.resolve_path loc xs in
  match decl with
  | Graph.NItem Ast.IExternDef _
  | Graph.NItem Ast.IDef _
  | Graph.NItem Ast.ITask _
  | Graph.NItem Ast.IVal _
  | Graph.NItem Ast.IMod _
  | Graph.NMethodDecl _ ->
      raise (Error.NamingError (loc, "Found non-type where type was expected"))
  | Graph.NItem IType (_, _, _, gs, _, _) ->
      let (ts, ctx) = lower_type_args xs ts gs ctx in
      (TNominal (xs, ts), ctx)
  | Graph.NItem Ast.IClass (_, _, _, gs, _, _)
  | Graph.NItem Ast.IExternType (_, _, _, gs, _) ->
      let (ts, ctx) = lower_type_args xs ts gs ctx in
      (TNominal (xs, ts), ctx)
  | _ ->
      unreachable ()

and lower_type_path loc xs ts ctx =
  match xs with
  | Ast.PRel [x] when ts = [] ->
      begin match ctx |> Ctx.find_gname_opt x with
      | Some x -> (TGeneric x, ctx)
      | None -> ctx |> lower_type_item_path loc xs ts
      end
  | _ -> ctx |> lower_type_item_path loc xs ts

and lower_expr_opt loc e ctx =
  match e with
  | Some e -> lower_expr e ctx
  | None -> ctx |> Ctx.new_expr (fun t -> ELit (loc, t, Ast.LUnit NoLoc))

and lower_mut loc e0 e1 ctx =
  let (e1, ctx) = ctx |> lower_expr e1 in
  match e0 with
  | Ast.EPath (loc, Ast.PRel [x], []) ->
      begin match ctx |> Ctx.find_vname loc x with
      | (x, Ctx.MVar) ->
          let (e0, ctx) = ctx |> Ctx.new_expr (fun t -> EVar (loc, t, x)) in
          ctx |> set_cell_expr loc e0 e1
      | (_, Ctx.MVal) -> raise (Error.NamingError (loc, "L-value is not a L-value"))
      end
  | Ast.ESelect (_, e00, e01) ->
      let (e00, ctx) = lower_expr e00 ctx in
      let (e01, ctx) = lower_expr e01 ctx in
      ctx |> replace_array_expr loc e00 e01 e1
  | Ast.EProject (_, e00, i) ->
      let (e00, ctx) = lower_expr e00 ctx in
      ctx |> Ctx.new_expr (fun t -> EUpdate (loc, t, e00, index_to_field i, e1))
  | Ast.EAccess (_, e00, x) ->
      let (e00, ctx) = lower_expr e00 ctx in
      ctx |> Ctx.new_expr (fun t -> EUpdate (loc, t, e00, x, e1))
  | _ -> panic "Expected variable, found path"

and lower_expr expr ctx =
  match expr with
  | Ast.EAnon _ ->
      let (x, ctx) = ctx |> Ctx.add_anon in
      ctx |> Ctx.new_expr (fun t -> EVar (NoLoc, t, x))
  | Ast.EBinOpRef (loc, op) ->
      let x = Ast.binop_name op in
      let xs = ["std"; indirect_name x] in
      ctx |> lower_expr_item_func loc xs [] []
  | Ast.EAccess (loc, e, x) ->
      let (e, ctx) = lower_expr e ctx in
      ctx |> Ctx.new_expr (fun t -> EAccess (loc, t, e, x))
  | Ast.EArray (loc, es, e) ->
      let (es, ctx) = es |> mapm lower_expr ctx in
      let (e0, ctx) = ctx |> make_array_expr loc es in
      begin match e with
      | None ->
          (e0, ctx)
      | Some e ->
          let (v1, ctx) = lower_expr e ctx in
          ctx |> append_array_expr loc e0 v1
      end
  | Ast.EBinOp (loc, Ast.BMut, _, e0, e1) ->
      lower_mut loc e0 e1 ctx
  | Ast.EBinOp (loc, Ast.BNotIn, ts, e0, e1) ->
      lower_expr (Ast.EUnOp (loc, Ast.UNot, [], (Ast.EBinOp (loc, Ast.BIn, ts, e0, e1)))) ctx
  | Ast.EBinOp (loc, Ast.BNeq, ts, e0, e1) ->
      lower_expr (Ast.EUnOp (loc, Ast.UNot, [], (Ast.EBinOp (loc, Ast.BEq, ts, e0, e1)))) ctx
  | Ast.EBinOp (loc, op, ts, e0, e1) ->
      let (x, ctx) = lower_binop op ctx in
      ctx |> lower_call true loc (Ast.EPath (loc, Ast.PRel [x], ts)) [e0; e1]
  | Ast.EUnOp (loc, op, ts, e) ->
      let (x, ctx) = lower_unop op ctx in
      ctx |> lower_call true loc (Ast.EPath (loc, Ast.PRel [x], ts)) [e]
  | Ast.ECall (loc, e, es) ->
      ctx |> lower_call false loc e es
  | Ast.EInvoke (loc, e, x, es) ->
      ctx |> lower_call false loc (Ast.EPath (loc, Ast.PRel [x], [])) ([e] @ es)
  | Ast.ECast (loc, e, t) ->
      let (e, ctx) = lower_expr e ctx in
      let (t, ctx) = lower_type t ctx in
      ctx |> Ctx.new_expr (fun t1 -> ECast (loc, t1, e, t))
  | Ast.EIf (loc, e, b0, b1) ->
      let (e, ctx) = lower_expr e ctx in
      let (b0, ctx) = lower_block b0 ctx in
      let (b1, ctx) = lower_block_opt b1 ctx in
      let (arm0, ctx) = ctx |> Ctx.new_pat (fun t -> (PConst (loc, t, Ast.LBool (loc, true)), b0)) in
      let (arm1, ctx) = ctx |> Ctx.new_pat (fun t -> (PIgnore (loc, t), b1)) in
      ctx |> Ctx.new_expr (fun t -> EMatch (loc, t, e, [arm0; arm1]))
  | Ast.ELit (_, l) ->
      lower_lit l ctx
  | Ast.ELoop (loc, b) ->
      let (b, ctx) = ctx |> lower_block b in
      ctx |> Ctx.new_expr (fun t -> ELoop (loc, t, b))
  | Ast.ESelect (loc, e0, e1) ->
      let (e0, ctx) = ctx |> lower_expr e0 in
      let (e1, ctx) = ctx |> lower_expr e1 in
      ctx |> get_array_expr loc e0 e1
  | Ast.ERecord (loc, (xes, e)) ->
      let (xes, ctx) = xes |> mapm (lower_field_expr loc) ctx in
      let (e, ctx) = ctx |> lower_expr_tail e in
      ctx |> Ctx.new_expr (fun t -> ERecord (loc, t, (xes, e)))
  | Ast.EEnwrap (loc, x, e) ->
      let (e, ctx) = ctx |> lower_expr e in
      ctx |> Ctx.new_expr (fun t -> EEnwrap (loc, t, x, e))
  | Ast.EReturn (loc, e) ->
      let (e, ctx) = lower_expr_opt loc e ctx in
      ctx |> Ctx.new_expr (fun t -> EReturn (loc, t, e))
  | Ast.EBreak (loc, e) ->
      let (e, ctx) = lower_expr_opt loc e ctx in
      ctx |> Ctx.new_expr (fun t -> EBreak (loc, t, e))
  | Ast.EContinue loc ->
      ctx |> Ctx.new_expr (fun t -> EContinue (loc, t))
  (* Desugared expressions *)
  | Ast.ETuple (loc, es) ->
      let (es, ctx) = es |> mapm lower_expr ctx in
      let xes = es |> indexes_to_fields in
      ctx |> Ctx.new_expr (fun t -> ERecord (loc, t, (xes, None)))
  | Ast.EProject (loc, e, i) ->
      let (e, ctx) = lower_expr e ctx in
      ctx |> Ctx.new_expr (fun t -> EAccess (loc, t, e, index_to_field i))
  | Ast.EBlock (_, b) ->
      let ((ss, e), ctx) = lower_block b ctx in
      let ctx = ctx |> Ctx.add_stmts ss in
      (e, ctx)
  | Ast.EFunc (loc, ps, e) ->
      lower_closure loc ps e ctx
  | Ast.ETask (_loc, _ps, _xts, _b) ->
      todo ()
  | Ast.EFor (_loc, _p, _e, _b) ->
      todo ()
  | Ast.EWhile (loc, e, b) ->
      let (e0, ctx) = ctx |> lower_expr e in
      (* Then-branch *)
      let (b0, ctx) = ctx |> lower_block b in
      (* Else-branch *)
      let (e1, ctx) = ctx |> Ctx.new_expr (fun t -> ELit (loc, t, Ast.LUnit loc)) in
      let (e1, ctx) = ctx |> Ctx.new_expr (fun t -> EBreak (loc, t, e1)) in
      let b1 = ([], e1) in
      (* If-stmt *)
      let (arm0, ctx) = ctx |> Ctx.new_pat (fun t -> (PConst (loc, t, Ast.LBool (loc, true)), b0)) in
      let (arm1, ctx) = ctx |> Ctx.new_pat (fun t -> (PIgnore (loc, t), b1)) in
      let (e2, ctx) = ctx |> Ctx.new_expr (fun t -> EMatch (loc, t, e0, [arm0; arm1])) in
      ctx |> Ctx.new_expr (fun t -> ELoop (loc, t, ([], e2)))
  | Ast.EWhileVal _ ->
      todo ()
  | Ast.EIfVal (loc, p, e, b0, b1) ->
      let ctx = ctx |> Ctx.push_vscope in
      let (e, ctx) = ctx |> lower_expr e in
      let (p0, ctx) = ctx |> lower_pat p in
      let (b0, ctx) = ctx |> lower_block b0 in
      let (b1, ctx) = ctx |> lower_block_opt b1 in
      let (es, ctx) = ctx |> Ctx.pop_vscope in
      let ctx = ctx |> Ctx.add_stmts es in
      let (p1, ctx) = ctx |> Ctx.new_pat (fun t -> (PIgnore (loc, t))) in
      let arms = [(p0, b0); (p1, b1)] in
      ctx |> Ctx.new_expr (fun t -> EMatch (loc, t, e, arms))
  | Ast.EMatch (loc, e, arms) ->
      let (v, ctx) = lower_expr e ctx in
      let (arms, ctx) = arms |> mapm lower_arm ctx in
      ctx |> Ctx.new_expr (fun t -> EMatch (loc, t, v, arms))
  | Ast.EOn _ ->
      todo ()
(*       let ctx = ctx |> Ctx.push_vscope in *)
(*       let (rs, ctx) = rs |> mapm lower_receiver ctx in *)
(*       let (ss, ctx) = ctx |> Ctx.pop_vscope in *)
(*       let ctx = ctx |> Ctx.add_stmts (ss |> rev) in *)
(*       (v2, ctx) *)
  | Ast.EPath (loc, xs, ts) ->
      ctx |> lower_expr_path loc xs ts
  | Ast.EFrom _ ->
      todo ()
  | Ast.EThrow _ -> todo ()
  | Ast.ETry _ -> todo ()

and lower_expr_tail t ctx =
  match t with
  | Some t ->
      let (t, ctx) = lower_expr t ctx in
      (Some t, ctx)
  | None ->
      (None, ctx)

and lower_pat_tail t ctx =
  match t with
  | Some t ->
      let (t, ctx) = lower_pat t ctx in
      (Some t, ctx)
  | None ->
      (None, ctx)

and lower_arm (p, e) ctx =
  let ctx = ctx |> Ctx.push_vscope in
  let (p, ctx) = lower_pat p ctx in
  let (e, ctx) = lower_expr e ctx in
  let (es, ctx) = ctx |> Ctx.pop_vscope in
  ((p, (es, e)), ctx)

(* and lower_receiver (p, e0, e1) ctx = *)
(*   let (v0, ctx) = lower_expr e0 ctx in *)
(*   let ctx = ctx |> Ctx.push_vscope in *)
(*   let (t, ctx) = ctx |> Ctx.fresh_t in *)
(*   let ctx = lower_irrefutable_pat p t v0 ctx in *)
(*   let (v1, ctx) = lower_expr e1 ctx in *)
(*   let (ss, ctx) = ctx |> Ctx.pop_vscope in *)
(*   (v0, (ss, v1), ctx) *)

and lower_unop op ctx =
  let x = Ast.unop_name op in
  (x, ctx)

and lower_field_expr loc (x, e) ctx =
  match e with
  | Some e ->
    let (e, ctx) = lower_expr e ctx in
    ((x, e), ctx)
  | None ->
      match ctx |> Ctx.find_vname_opt x with
      | Some (v, MVal) ->
          let (e, ctx) = ctx |> Ctx.new_expr (fun t -> EVar (loc, t, v)) in
          ((x, e), ctx)
      | Some (e, MVar) ->
          let (e, ctx) = ctx |> Ctx.new_expr (fun t -> EVar (loc, t, e)) in
          let (v, ctx) = get_cell_expr loc e ctx in
          ((x, v), ctx)
      | None -> panic "Name not found"

and lower_field_type (x, t) ctx =
  match t with
  | Some t ->
    let (t, ctx) = lower_type t ctx in
    ((x, t), ctx)
  | None ->
    let (t, ctx) = ctx |> Ctx.fresh_t in
    ((x, t), ctx)

and lower_variant_type (x, t) ctx =
  let (t, ctx) = lower_type t ctx in
  ((x, t), ctx)

and lower_type_or_fresh t ctx =
  match t with
  | Some t -> lower_type t ctx
  | None -> ctx |> Ctx.fresh_t

and lower_type_or_unit t ctx =
  match t with
  | Some t -> lower_type t ctx
  | None -> (atom "unit", ctx)

and lower_type t ctx =
  match t with
  | Ast.TFunc (loc, ts, t) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = lower_type t ctx in
      let (tr, ctx) = empty_type_env loc ctx in
      let tf = TFunc (ts @ [tr], t) in
      let tc = TRecord ([("r", tr); ("f", tf)] |> fields_to_rows TRowEmpty) in
      (tc, ctx)
  | Ast.TTuple (_, ts) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let t = types_to_record ts in
      (t, ctx)
  | Ast.TRecord (_, (fs, t)) ->
      let (fs, ctx) = fs |> mapm lower_field_type ctx in
      let (t, ctx) = match t with
      | Some t -> lower_type t ctx
      | None -> (TRowEmpty, ctx)
      in
      let t = TRecord (fs |> fields_to_rows t) in
      (t, ctx)
  | Ast.TEnum (_, (vs, t)) ->
      let (vs, ctx) = vs |> mapm lower_variant_type ctx in
      let (t, ctx) = match t with
      | Some t -> lower_type t ctx
      | None -> (TRowEmpty, ctx)
      in
      let t = TEnum (vs |> fields_to_rows t) in
      (t, ctx)
  | Ast.TPath (loc, xs, ts) ->
      ctx |> lower_type_path loc xs ts
  | Ast.TArray (_, t) ->
      let (t, ctx) = lower_type t ctx in
      (nominal "Array" [t], ctx)

and lower_pat p ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  ctx |> lower_typed_pat p t

(* Lowers an irrefutable pattern matching on variable v, e.g., val p = v; *)
and lower_typed_pat p t ctx =
  match p with
  | Ast.PIgnore loc ->
      (PIgnore (loc, t), ctx)
  | Ast.PRecord (loc, (xps, p)) ->
      let (xps, ctx) = xps |> mapm (fun (x, p) ctx ->
        match p with
        | Some p ->
            let (p, ctx) = lower_pat p ctx in
            ((x, p), ctx)
        | None ->
            let (x, ctx) = ctx |> Ctx.bind_vname x MVal in
            let (p, ctx) = ctx |> Ctx.new_pat (fun t -> PVar (loc, t, x)) in
            ((x, p), ctx)
      ) ctx in
      let (p, ctx) = ctx |> lower_pat_tail p in
      (PRecord (loc, t, (xps, p)), ctx)
  | Ast.PTuple (loc, ps) ->
      let xps = ps |> map (fun p -> Some p) |> indexes_to_fields in
      let p = (Ast.PRecord (loc, (xps, None))) in
      lower_typed_pat p t ctx
  | Ast.PArray (_loc, _ps, _p) ->
      todo()
  | Ast.PVar (loc, x) ->
      let (x, ctx) = ctx |> Ctx.bind_vname x MVal in
      (PVar (loc, t, x), ctx)
  | Ast.PUnwrap (loc, x, p) ->
      let (p, ctx) = lower_pat p ctx in
      (PUnwrap (loc, t, x, p), ctx)
  | Ast.POr (loc, p0, p1) ->
      let (p0, ctx) = lower_pat p0 ctx in
      let (p1, ctx) = lower_pat p1 ctx in
      (POr (loc, t, p0, p1), ctx)
  | Ast.PConst (loc, l) ->
      (PConst (loc, t, l), ctx)

and lower_block (ss, expr) ctx =
  let rec lower_stmts ss ctx =
    match ss with
    | s::ss ->
        begin match s with
        | Ast.SNoop _ ->
            ctx |> lower_stmts ss
        | Ast.SVal (loc, (p, t), e0) ->
            let (e0, ctx) = ctx |> lower_expr e0 in
            let (t, ctx) = ctx |> lower_type_or_fresh t in
            let (p, ctx) = ctx |> lower_typed_pat p t in
            let ctx = ctx |> Ctx.push_vscope in
            let (e1, ctx) = ctx |> lower_stmts ss in
            let (es, ctx) = ctx |> Ctx.pop_vscope in
            ctx |> Ctx.new_expr (fun t1 -> EMatch (loc, t1, e0, [(p, (es, e1))]))
        | Ast.SVar (loc, (x, t), e0) ->
            let (e0, ctx) = ctx |> lower_expr e0 in
            let (e0, ctx) = ctx |> new_cell_expr loc e0 in
            let (t, ctx) = ctx |> lower_type_or_fresh t in
            let t = cell_type t in
            let (x, ctx) = ctx |> Ctx.bind_vname x MVar in
            let p = PVar (loc, t, x) in
            let ctx = ctx |> Ctx.push_vscope in
            let (e1, ctx) = ctx |> lower_stmts ss in
            let (es, ctx) = ctx |> Ctx.pop_vscope in
            ctx |> Ctx.new_expr (fun t1 -> EMatch (loc, t1, e0, [(p, (es, e1))]))
        | Ast.SExpr (loc, e) ->
            begin match e with
            | Ast.EReturn _ | Ast.EBreak _ | Ast.EContinue _ ->
                begin match (ss, expr) with
                | ([], None) -> ctx |> lower_expr e
                | _ -> raise (Error.TypingError (loc, "Found unreachable code beyond this point"))
                end
            | _ ->
              let (e, ctx) = ctx |> lower_expr e in
              let ctx = ctx |> Ctx.add_stmt_expr e in
              ctx |> lower_stmts ss
            end
        end
    | [] ->
        begin match expr with
        | Some e -> ctx |> lower_expr e
        | None -> ctx |> Ctx.new_expr (fun t -> ELit (NoLoc, t, Ast.LUnit NoLoc))
        end
  in
  let ctx = ctx |> Ctx.push_vscope in
  let (e, ctx) = ctx |> lower_stmts ss in
  let (es, ctx) = ctx |> Ctx.pop_vscope in
  ((es, e), ctx)

and lower_block_opt b ctx =
  match b with
  | Some b ->
      ctx |> lower_block b
  | None ->
      ctx |> empty_block Error.NoLoc

and lower_binop op ctx =
  let x = Ast.binop_name op in
  (x, ctx)

and splice_regex = (Str.regexp "\\${[^}]+}\\|\\$[a-zA-Z_][a-zA-Z0-9_]*")

and str_to_string loc s ctx =
  let (e0, ctx) = ctx |> Ctx.new_expr (fun t -> ELit (loc, t, Ast.LString (loc, s))) in
  let (e1, ctx) = ctx |> Ctx.new_expr (fun t -> EItem (loc, t, ["std"; "from_str"], [])) in
  let (v, ctx) = ctx |> Ctx.new_expr (fun t -> ECallExpr (loc, t, e1, [e0])) in
  (v, ctx)

and lower_lit l ctx =
  match l with
  (* Lower interpolated string literals *)
  | Ast.LString (loc, s) ->
      let (es, ctx) = s |> Str.full_split splice_regex
        |> mapm (fun s ctx ->
          match s with
          | Str.Text s ->
              str_to_string loc s ctx
          | Str.Delim s ->
              let s = String.sub s 1 ((String.length s) - 1) in
              let e = Parser.expr Lexer.main (Lexing.from_string s) in
              let ctx = ctx |> Ctx.push_vscope in
              let (e, ctx) = ctx |> lower_expr e in
              let (es, ctx) = ctx |> Ctx.pop_vscope in
              let ctx = ctx |> Ctx.add_stmts es in
              (e, ctx)
        ) ctx in
      begin match es with
      | v::vs ->
        vs |> foldl (fun (v1, ctx) v2 ->
          let (v0, ctx) = ctx |> Ctx.new_expr (fun t -> EItem (loc, t, ["std"; "concat"], [])) in
          ctx |> Ctx.new_expr (fun t -> ECallExpr (loc, t, v0, [v1; v2]))
        ) (v, ctx)
      | [] -> str_to_string loc s ctx
      end
  | _ -> ctx |> Ctx.new_expr (fun t -> ELit (NoLoc, t, l))

and lower_closure loc pts b ctx =

  (* Print.Ast.pr_block b Print.Ctx.brief; *)

  (* Compile the closure body *)
  let ctx = ctx |> Ctx.push_vscope in
  let (pts, ctx) = pts |> mapm lower_param ctx in
  let ((ss0, e), ctx) = lower_block b ctx in
  let (ss1, ctx) = ctx |> Ctx.pop_vscope in
  let b = (ss1 @ ss0, e) in
  (* Print.Ir1.pr_block b Print.Ctx.brief; *)

  (* Create the variables which need to be stored in the environment *)
  let fvs = free_vars (pts |> map fst |> bound_vars) b in

  (* Create an extra function parameter for the environment *)
  let (v_env, ctx) = ctx |> Ctx.fresh_x in
  let (t_env, ctx) = ctx |> Ctx.fresh_t in

  (* Create the function signature *)
  let ts = pts |> map snd in
  let (vs, ctx) = pts |> mapm (fun _ ctx -> Ctx.fresh_x ctx) ctx in
  let vts = (zip vs ts) @ [(v_env, t_env)] in

  (* Create code for unpacking the environment inside the closure *)
  let (es, ctx) = ctx |> vars_to_exprs loc vs in
  let (e_env, ctx) = ctx |> Ctx.new_expr (fun t -> EVar (loc, t, v_env)) in
  let (e, ctx) = ctx |> exprs_to_record loc (es @ [e_env]) in

  let ps = pts |> map fst in
  let (ps_env, ctx) = ctx |> vars_to_patterns loc fvs in
  let (p_env, ctx) = ctx |> patterns_to_record loc ps_env in
  let (p, ctx) = ctx |> patterns_to_record loc (ps @ [p_env]) in

  let (e, ctx) = ctx |> Ctx.new_expr (fun t -> EMatch (loc, t, e, [(p, b)])) in

  (* Create the function *)
  let (xs, ctx) = ctx |> Ctx.fresh_f in
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let ctx = ctx |> Ctx.add_item xs (IDef (loc, [], [], vts, t, [], ([], e))) in

  (* Create the function pointer *)
  let (e_fun, ctx) = ctx |> Ctx.new_expr (fun t -> EItem (loc, t, xs, [])) in

  (* Printf.printf "\n"; *)
  (* Printf.printf "B-vars: "; *)
  (* bvs |> List.iter (fun v -> Printf.printf "%s, " v); *)
  (* Printf.printf "\n"; *)
  (* Printf.printf "F-vars: "; *)
  (* fvs |> List.iter (fun v -> Printf.printf "%s, " v); *)
  (* Printf.printf "\n"; *)
  let (es, ctx) = ctx |> vars_to_exprs loc fvs in
  let (e_env, ctx) = ctx |> exprs_to_record loc es in
  ctx |> Ctx.new_expr (fun t -> ERecord (loc, t, ([("f", e_fun); ("r", e_env)], None)))

(* Create a new cell *)
and new_cell_expr loc e ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "new_cell"], [t], [e]))

and cell_type t =
  nominal "Cell" [t]

(* Retrieve the value from a cell *)
and get_cell_expr loc e ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "get_cell"], [t], [e]))

(* Update the value inside a cell *)
and set_cell_expr loc e0 e1 ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "set_cell"], [t], [e0; e1]))

(* Create an empty block *)
and empty_block loc ctx =
  let (v, ctx) = ctx |> Ctx.new_expr (fun t -> ELit (loc, t, Ast.LUnit loc)) in
  (([], v), ctx)

and empty_expr_env loc ctx =
  lower_expr (Ast.ERecord (loc, ([], None))) ctx

and empty_type_env loc ctx =
  lower_type (Ast.TRecord (loc, ([], None))) ctx

and make_array_expr loc es ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let (v0, ctx) = ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "array"], [t], [])) in
  let ctx = es |> foldl (fun ctx v1 -> ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "push_back"], [t], [v0; v1])) |> snd) ctx in
  (v0, ctx)

and append_array_expr loc e0 e1 ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "append"], [t], [e0; e1]))

and get_array_expr loc e0 e1 ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "get"], [t], [e0; e1]))

and replace_array_expr loc e0 e1 e2 ctx =
  let (t, ctx) = ctx |> Ctx.fresh_t in
  ctx |> Ctx.new_expr (fun t1 -> ECallItem (loc, t1, ["std"; "replace"], [t], [e0; e1; e2]))
