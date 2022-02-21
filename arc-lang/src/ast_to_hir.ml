open Hir
open Table
open Utils

type clause = eq list * subst list * Ast.expr
and subst = (name * name)
and eq = name * Ast.pattern

module Ctx = struct
  type t = {
    table: Table.table;         (* Table of top-level item declarations *)
    next_def_uid: Gen.t;        (* Function uid counter *)
    next_type_uid: Gen.t;       (* Type uid counter *)
    next_generic_uid: Gen.t;    (* Generic uid counter *)
    next_expr_uid: Gen.t;       (* Expression uid counter *)
    vstack: vscope list;        (* Stack of scopes for value parameters *)
    gstack: gscope list;        (* Stack of scopes for type parameters *)
    path: path;                 (* Current path *)
    hir: definition list;       (* Final output of the lowering *)
    then_clauses: clause list;  (* Then-clauses created during pattern compilation *)
    else_clauses: clause list;  (* Else-clauses created during pattern compilation *)
    astack: ascope list;   (* Stack of anonymous variables *)
  }
  and vscope = {
    vsubst: (name * (name * mut)) list;
    stmts: Hir.ssa list;
  }
  and mut =
    | MVar
    | MVal
  and gscope = {
    gsubst: (name * name) list;
  }
  and ascope = {
    avars: name list;
  }
  and definition = path * item

  let rec make table = {
    table = table;
    next_def_uid = Gen.make ();
    next_type_uid = Gen.make ();
    next_generic_uid = Gen.make ();
    next_expr_uid = Gen.make ();
    vstack = [];
    gstack = [];
    astack = [];
    path = [];
    hir = [];
    then_clauses = [];
    else_clauses = [];
  }


  and pr_subst (x, (a, _)) ctx =
    Pretty.pr_name x ctx;
    Pretty.pr " => ";
    Pretty.pr_name a ctx

  and print_substs substs ctx =
    Pretty.pr "Substs: ";
    Pretty.pr_brack (Pretty.pr_list pr_subst substs) ctx

  and print_scopes scopes ctx =
    match scopes with
    | [] -> ()
    | scope::scopes ->
        Pretty.pr "Scope:";
        let ctx = ctx |> Pretty.Ctx.indent in
        ctx |> Pretty.pr_indent;
        print_substs scope.vsubst ctx;
        ctx |> Pretty.pr_indent;
        print_scopes scopes ctx;

  and add_item xs i (ctx:t) =
    { ctx with hir = (xs, i)::ctx.hir }

  and fresh_ts n (ctx:t) =
    repeat fresh_t n ctx

  and fresh_t (ctx:t) =
    let (n, next_type_uid) = ctx.next_type_uid |> Gen.fresh in
    let ctx = { ctx with next_type_uid } in
    let x = Printf.sprintf "%d" n in
    let t = Hir.TVar x in
    (t, ctx)

  and fresh_x (ctx:t) =
    let (n, next_expr_uid) = ctx.next_expr_uid |> Gen.fresh in
    let x = Printf.sprintf "x%d" n in
    let ctx = { ctx with next_expr_uid; } in
    (x, ctx)

  and fresh_f (ctx:t) =
    let (n, next_def_uid) = ctx.next_def_uid |> Gen.fresh in
    let x = Printf.sprintf "f%d" n in
    let ctx = { ctx with next_def_uid; } in
    ([x], ctx)

  and fresh_g (ctx:t) =
    let (n, next_generic_uid) = ctx.next_generic_uid |> Gen.fresh in
    let g = Printf.sprintf "T%d" n in
    let ctx = { ctx with next_generic_uid; } in
    (g, ctx)

  and add_expr e (ctx:t) = 
    let (t, ctx) = ctx |> fresh_t in
    ctx |> add_typed_expr e t

  and add_named_expr e x (ctx:t) =
    let (t, ctx) = ctx |> fresh_t in
    let s = (x, t, e) in
    let ctx = ctx |> add_stmt s in
    (x, ctx)
    
  and add_typed_expr e t (ctx:t) = 
    let (x, ctx) = ctx |> fresh_x in
    let s = (x, t, e) in
    let ctx = ctx |> add_stmt s in
    (x, ctx)

  and add_stmt (s:Hir.ssa) (ctx:t) =
    match ctx.vstack with
    | h::t -> { ctx with vstack = { h with stmts = s::h.stmts}::t }
    | [] -> unreachable ()

  and add_stmts s (ctx:t) =
    match ctx.vstack with
    | h::t -> { ctx with vstack = { h with stmts = s @ h.stmts}::t }
    | [] -> unreachable ()

  and push_vscope (ctx:t) =
    { ctx with vstack = { stmts = []; vsubst = [] }::ctx.vstack }

  and pop_vscope (ctx:t) =
    match ctx.vstack with
    | {stmts; _}::vstack -> (stmts |> List.rev, { ctx with vstack })
    | [] -> unreachable ()

  and push_gscope (ctx:t) =
    { ctx with gstack = { gsubst = [] }::ctx.gstack }

  and pop_gscope (ctx:t) =
    { ctx with gstack = ctx.gstack |> List.tl }

  and push_ascope (ctx:t) =
    { ctx with astack = { avars = [] }::ctx.astack }

  and pop_ascope (ctx:t) =
    let vs = (ctx.astack |> List.hd).avars in
    let ctx = { ctx with astack = ctx.astack |> List.tl } in
    (vs, ctx)

  and add_anon (ctx:t) =
    let (x, ctx) = ctx |> fresh_x in
    match ctx.astack with
    | h::t ->
        let ctx = { ctx with astack = { avars = x::h.avars}::t } in
        (x, ctx)
    | [] -> unreachable ()

  (* Returns a name path *)
  and item_path x ctx =
    x::ctx.path |> List.rev

  and bind_vname v m (ctx:t) =
    match ctx.vstack with
    | hd::tl ->
        let (v', ctx) = ctx |> fresh_x in
        let vstack = { hd with vsubst = (v, (v', m))::hd.vsubst }::tl in
        let ctx = { ctx with vstack } in
        (v', ctx)
    | [] -> unreachable ()

  and bind_gname g (ctx:t) =
    match ctx.gstack with
    | hd::tl ->
        let (g', ctx) = ctx |> fresh_g in
        let gstack = { gsubst = (g, g')::hd.gsubst}::tl in
        let ctx = { ctx with gstack } in
        (g', ctx)
    | [] -> unreachable ()

  and rename_vname v v' (ctx:t) =
    match ctx.vstack with
    | hd::tl ->
        let hd = { hd with vsubst = (v, v')::hd.vsubst } in
        { ctx with vstack = hd::tl }
    | [] -> unreachable ()

  (* Finds a value variable *)
  and find_vname v (ctx:t) =
    ctx.vstack |> List.find_map (fun vscope -> vscope.vsubst |> List.assoc_opt v)

  (* Finds a generic variable *)
  and find_gname g (ctx:t) =
    ctx.gstack |> List.find_map (fun gscope -> gscope.gsubst |> List.assoc_opt g)

  and push_namespace name ctx =
    let ctx = { ctx with path = name::ctx.path } in
    (ctx.path |> List.rev, ctx)

  (* Annotates SSA with name v in current scope to have type t0 *)
  and annotate v t0 ctx =
    match ctx.vstack with
    | [] -> unreachable ()
    | hd0::tl0 ->
        let stmts = hd0.stmts |> List.map (fun (x, t1, e) ->
          let t = if v = x then t0 else t1 in
          (x, t, e)
        ) in
        { ctx with vstack = { hd0 with stmts}::tl0 }

  and pop_namespace ctx = { ctx with path = ctx.path |> List.tl }

  and add_then_clause c ctx = { ctx with then_clauses = c::ctx.then_clauses }

  and add_else_clause c ctx = { ctx with else_clauses = c::ctx.else_clauses }

  and take_clauses ctx =
    let then_clauses = ctx.then_clauses |> List.rev in
    let else_clauses = ctx.else_clauses |> List.rev in
    let ctx = { ctx with then_clauses = []; else_clauses = [] } in
    (then_clauses, else_clauses, ctx)

  and resolve_path path (ctx:t) =
    let rec resolve_path xs (ctx:t) =
      match ctx.table |> PathMap.find_opt xs with
      | Some (DUse xs) -> ctx |> resolve_path xs
      | Some (DItem decl) -> (xs, decl)
      | None -> panic (Printf.sprintf "Path not found \"%s\"" (Pretty.path_to_str xs))
    in ctx |> resolve_path (path @ ctx.path)

  and resolve_type_path xs ts ctx =
    let resolve_type_path xs ctx =
      let (xs, decl) = ctx |> resolve_path xs in
      match decl with
      | DExternDef _ | DDef _ | DTask _ | DVariant _ | DGlobal | DMod -> unreachable ()
      | DTypeAlias (n, _gs, _t) ->
          let (_ts, _ctx) = match List.length ts with
          | m when m = 0 -> fresh_ts n ctx
          | m when m = n -> (ts, ctx)
          | m -> panic (Printf.sprintf "Type alias \"%s\" expects %d arguments, but %d were given" (Pretty.path_to_str xs) n m)
          in
          todo ()
      | DEnum n | DClass n | DExternType n ->
          let (ts, ctx) = match List.length ts with
          | m when m = 0 -> fresh_ts n ctx
          | m when m = n -> (ts, ctx)
          | m -> panic (Printf.sprintf "Type path \"%s\" has wrong number of type arguments, expected %d but found %d" (Pretty.path_to_str xs) n m)
          in
          (Hir.TNominal (xs, ts), ctx)
    in
    match xs with
    | [x] when ts = [] ->
        begin match ctx |> find_gname x with
        | Some x -> (Hir.TGeneric x, ctx)
        | None -> ctx |> resolve_type_path xs
        end
    | _ -> ctx |> resolve_type_path xs

  (* Resolves a path expression *)
  and resolve_expr_path xs ts ctx =
    let resolve_expr_path xs ctx =
      let (xs, decl) = ctx |> resolve_path xs in
      match decl with
      | DEnum n | DClass n | DExternDef n | DExternType n | DTask n | DDef n ->
          begin
            let (ts, ctx) = match List.length ts with
            | m when m = 0 -> fresh_ts n ctx
            | m when m = n -> (ts, ctx)
            | m -> panic (Printf.sprintf "Type path \"%s\" has wrong number of type arguments, expected %d but found %d" 
                (Pretty.path_to_str xs) n m
              )
            in
            ctx |> add_expr (Hir.EItem (xs, ts))
          end
      | DGlobal ->
          begin
            match List.length ts with
            | 0 -> ctx |> add_expr (Hir.EItem (xs, []))
            | _ -> panic (Printf.sprintf "Path \"%s\" has type arguments" (Pretty.path_to_str xs))
          end
      | DMod | DVariant _ | DTypeAlias _ -> panic "Found non-expr item where expr was expected"
    in
    match xs with
    | [x] when ts = [] ->
        begin match ctx |> find_vname x with
        | Some (v, MVal) -> (v, ctx)
        | Some (v, MVar) -> read_cell v ctx
        | None -> ctx |> resolve_expr_path xs
        end
    | _ -> ctx |> resolve_expr_path xs

  and resolve_lvalue e ctx =
    match e with
    | Ast.EPath ([x], []) ->
        (* L-value must be a cell. TODO: Support more L-values *)
        begin match ctx |> find_vname x with
        | Some (v, MVar) -> (v, ctx)
        | Some (_, MVal) -> panic "L-value is a value"
        | None -> panic "Variable not bound"
        end
    | _ -> panic "Expected variable, found path"

  (* Returns set of currently visible variables *)
  and visible ctx =
    let rec visible vstack acc =
    match vstack with
    | h::t -> h.stmts |> List.fold_left (fun acc (v, _, _) -> v::acc) acc |> visible t
    | [] -> acc |> List.rev
    in
    visible ctx.vstack [] 

  (* Create a new cell *)
  and new_cell v t ctx =
    let (v_fun, ctx) = ctx |> add_expr (Hir.EItem (["cell"], [t])) in
    ctx |> add_expr (Hir.ECall (v_fun, [v]))

  (* Retrieve the value from a cell *)
  and read_cell v ctx =
    let (t, ctx) = ctx |> fresh_t in
    let (v_fun, ctx) = ctx |> add_expr (Hir.EItem (["read"], [t])) in
    ctx |> add_expr (Hir.ECall (v_fun, [v]))

  (* Update the value inside a cell *)
  and update_cell v0 v1 ctx =
    let (t, ctx) = ctx |> fresh_t in
    let (v_fun, ctx) = ctx |> add_expr (Hir.EItem (["update"], [t])) in
    ctx |> add_expr (Hir.ECall (v_fun, [v0; v1]))

  (* Create an empty block *)
  and empty_block ctx =
    let ctx = ctx |> push_vscope in
    let (v, ctx) = ctx |> add_expr (Hir.ELit Ast.LUnit) in
    let (ss, ctx) = ctx |> pop_vscope in
    ((ss, v), ctx)

  and make_array vs ctx =
    let (t, ctx) = ctx |> fresh_t in
    let (v_new, ctx) = ctx |> add_expr (Hir.EItem (["array"], [t])) in
    let (v_push, ctx) = ctx |> add_expr (Hir.EItem (["push"], [t])) in
    let (v0, ctx) = ctx |> add_expr (Hir.ECall (v_new, [])) in
    let ctx = vs |> foldl (fun ctx v1 -> ctx |> add_expr (Hir.ECall (v_push, [v0; v1])) |> snd) ctx in
    (v0, ctx)

  and append_array v0 v1 ctx =
    let (t, ctx) = ctx |> fresh_t in
    let (v_append, ctx) = ctx |> add_expr (Hir.EItem (["append"], [t])) in
    ctx |> add_expr (Hir.ECall (v_append, [v0; v1]))

  (* Retrieve the value from a cell *)
  and select_array v0 v1 ctx =
    let (t, ctx) = ctx |> fresh_t in
    let (v_fun, ctx) = ctx |> add_expr (Hir.EItem (["select"], [t])) in
    ctx |> add_expr (Hir.ECall (v_fun, [v0; v1]))

  and nominal x ts = Hir.TNominal ([x], ts)

  and generic x = Hir.TGeneric x
end

let rec hir_of_ast table ast =
  let ctx = Ctx.make table in
  let ctx = ast |> List.fold_left (fun ctx i -> lower_item i ctx) ctx in
  let hir = ctx.hir |> List.rev in
  hir

and lower_item i ctx =
  match i with
  | Ast.IVal (d, x, t, e) ->
      let xs = ctx |> Ctx.item_path x in
      let (t, ctx) = lower_type_or_fresh t ctx in
      let ctx = ctx |> Ctx.push_vscope in
      let (v, ctx) = lower_expr e ctx in
      let (ss, ctx) = ctx |> Ctx.pop_vscope in
      let b = (ss, v) in
      ctx |> Ctx.add_item xs (Hir.IVal (d, t, b))
  | Ast.IEnum (d, x, gs, variants) ->
      let xs = x::ctx.path |> List.rev in
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (variants, ctx) = variants |> mapm (lower_variant xs) ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (Hir.IEnum (d, gs, variants))
  | Ast.IExternDef (d, x, gs, ts, t) ->
      let x = Ast.def_name x in
      let xs = x::ctx.path |> List.rev in
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = lower_type_or_unit t ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (Hir.IExternDef (d, gs, ts, t))
  | Ast.IExternType (d, x, gs) ->
      let xs = x::ctx.path |> List.rev in
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (Hir.IExternType (d, gs))
  | Ast.IClass (d, x, gs, decls) ->
      let xs = x::ctx.path |> List.rev in
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let ctx = decls |> foldl (lower_decl xs) ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (Hir.IClass (d, gs))
  | Ast.IInstance (d, gs, xs, ts, defs) ->
      let ctx = ctx |> Ctx.push_gscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let ctx = defs |> foldl (lower_def xs) ctx in
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (Hir.IInstance (d, gs, xs, ts))
  (* Declaration *)
  | Ast.IDef (_, _, _, _, _, None) -> ctx
  (* Definition *)
  | Ast.IDef (d, x, gs, ps, t, Some b) ->
      let x = Ast.def_name x in
      let xs = x::ctx.path in
      let ctx = ctx |> Ctx.push_gscope in
      let ctx = ctx |> Ctx.push_vscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (ps, ctx) = ps |> mapm lower_param ctx in
      let (t, ctx) = lower_type_or_fresh t ctx in
      let ((ss0, v), ctx) = lower_block b ctx in
      let (ss1, ctx) = ctx |> Ctx.pop_vscope in
      let b = (ss1 @ ss0, v) in
      let ctx = ctx |> Ctx.pop_gscope in
      ctx |> Ctx.add_item xs (Hir.IDef (d, gs, ps, t, b))
  | Ast.ITask (_, _, _, _, _, None) -> ctx
  | Ast.ITask (d, x, gs, ps0, ps1, Some b) ->
      let x = Ast.def_name x in
      let xs = x::ctx.path in
      let ctx = ctx |> Ctx.push_gscope in
      let ctx = ctx |> Ctx.push_vscope in
      let (gs, ctx) = gs |> mapm lower_generic ctx in
      let (ps0, ctx) = ps0 |> mapm lower_param ctx in
      let (ps1, ctx) = ps1 |> mapm lower_param ctx in

      let ((ss0, v), ctx) = lower_block b ctx in

      let (ss1, ctx) = ctx |> Ctx.pop_vscope in
      let ctx = ctx |> Ctx.pop_gscope in
      let b = (ss1 @ ss0, v) in
      ctx |> Ctx.add_item xs (Hir.ITask (d, gs, ps0, ps1, b))
  | Ast.ITypeAlias _ ->
      todo ()
  | Ast.IMod (_, x, is) ->
      let (_, ctx) = ctx |> Ctx.push_namespace x in
      let ctx = is |> List.fold_left (fun ctx i -> lower_item i ctx) ctx in
      let ctx = ctx |> Ctx.pop_namespace in
      ctx
  | Ast.IUse _ -> ctx

and lower_decl _xs _ctx (_d:Ast.decl) =
  todo ()

and lower_def _xs _ctx (_d:Ast.def) =
  todo ()

and lower_variant xs (x, ts) (ctx:Ctx.t) =
  let xs = x::xs |> List.rev in
  let (t, ctx) = match ts with
  | [] -> (unit_type, ctx)
  | [t] -> lower_type t ctx
  | ts ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let fts = ts |> Hir.indexes_to_fields in
      let t = fts |> fields_to_rows Hir.TRowEmpty in
      (Hir.TRecord t, ctx)
  in
  (xs, ctx |> Ctx.add_item xs (Hir.IVariant t))

and lower_generic x (ctx:Ctx.t) =
  let (x, ctx) = ctx |> Ctx.bind_gname x in
  (x, ctx)

and lower_param (p, t) (ctx:Ctx.t) =
  let (x', ctx) = ctx |> Ctx.fresh_x in
  let (t, ctx) = lower_type_or_fresh t ctx in
  let ctx = match p with
  | Ast.PVar x -> ctx |> Ctx.rename_vname x (x', MVal)
  | _ -> lower_irrefutable_pat p t x' ctx
  in
  ((x', t), ctx)

and lower_arg_expr (e:Ast.expr) ctx =
  let ctx = ctx |> Ctx.push_ascope in
  let ctx = ctx |> Ctx.push_vscope in
  let (v, ctx) = lower_expr e ctx in
  let (ss, ctx) = ctx |> Ctx.pop_vscope in
  let (vs, ctx) = ctx |> Ctx.pop_ascope in
  match vs with
  | [] ->
      let ctx = ctx |> Ctx.add_stmts ss in
      (v, ctx)
  | vs ->
      let (ps, ctx) = vs |> mapm (fun v ctx ->
          let (t, ctx) = ctx |> Ctx.fresh_t in
          ((v, t), ctx)
      ) ctx in
      let (xs, ctx) = ctx |> Ctx.fresh_f in
      let (t, ctx) = ctx |> Ctx.fresh_t in
      let b = (ss, v) in
      let i = Hir.IDef ([], [], ps, t, b) in
      let ctx = ctx |> Ctx.add_item xs i in
      ctx |> Ctx.add_expr (Hir.EItem (xs, []))


and lower_call e es ctx =
  let lower_call e (es:Ast.expr list) ctx =
    let (v, ctx) = lower_expr e ctx in
    let (vs, ctx) = es |> mapm lower_arg_expr ctx in
    ctx |> Ctx.add_expr (Hir.ECall (v, vs))
  in
  match e with
  | Ast.EPath (xs, ts) ->
      let resolve_call_path xs ctx =
        begin match ctx |> Ctx.resolve_path xs with
        | (xs, DVariant n) ->
            let (ts, ctx) = match List.length ts with
            | m when m = n -> ts |> mapm lower_type ctx
            | m when m = 0 -> ctx |> Ctx.fresh_ts n
            | m -> panic (Printf.sprintf "Variant path \"%s\" has wrong number of type arguments, expected %d but found %d" (Pretty.path_to_str xs) n m)
            in
            let (v, ctx) =
              match es with
              | [] ->
                  ctx |> Ctx.add_expr (Hir.ELit (Ast.LUnit))
              | [e] ->
                  lower_expr e ctx
              | es ->
                  let (vs, ctx) = es |> mapm lower_expr ctx in
                  let fs = vs |> Hir.indexes_to_fields in
                  ctx |> Ctx.add_expr (Hir.ERecord fs)
            in
            ctx |> Ctx.add_expr (Hir.EEnwrap (xs, ts, v))
        | _ -> lower_call e es ctx
        end
      in
      begin match xs with
      | [x] when ts = [] ->
          begin match ctx |> Ctx.find_vname x with
          | Some (_, Ctx.MVal) -> lower_call e es ctx
          | Some (v, Ctx.MVar) -> Ctx.read_cell v ctx
          | None -> ctx |> resolve_call_path xs
          end
      | _ -> ctx |> resolve_call_path xs
      end
  | _ -> lower_call e es ctx

and lower_expr_opt e (ctx:Ctx.t) =
  match e with
  | Some e -> lower_expr e ctx
  | None -> ctx |> Ctx.add_expr (Hir.ELit Ast.LUnit)

and lower_expr expr ctx =
  match expr with
  | Ast.EAnon ->
      ctx |> Ctx.add_anon
  | Ast.EBinOpRef op ->
      let x = Ast.binop_name op in
      ctx |> Ctx.add_expr (Hir.EItem ([x], []))
  | Ast.EAccess (e, x) -> 
      let (v, ctx) = lower_expr e ctx in
      ctx |> Ctx.add_expr (Hir.EAccess (v, x))
  | Ast.EArray (es, e) ->
      let (vs, ctx) = es |> mapm lower_expr ctx in
      let (v0, ctx) = ctx |> Ctx.make_array vs in
      begin match e with
      | None ->
          (v0, ctx)
      | Some e ->
          let (v1, ctx) = lower_expr e ctx in
          ctx |> Ctx.append_array v0 v1
      end
  | Ast.EBinOp (Ast.BMut, e0, e1) ->
      let (v0, ctx) = Ctx.resolve_lvalue e0 ctx in
      let (v1, ctx) = lower_expr e1 ctx in
      ctx |> Ctx.update_cell v0 v1
  | Ast.EBinOp (Ast.BNotIn, e0, e1) ->
      lower_expr (Ast.EUnOp (Ast.UNot, (Ast.EBinOp (Ast.BIn, e0, e1)))) ctx
  | Ast.EBinOp (Ast.BNeq, e0, e1) ->
      lower_expr (Ast.EUnOp (Ast.UNot, (Ast.EBinOp (Ast.BEq, e0, e1)))) ctx
  | Ast.EBinOp (op, e0, e1) ->
      let (v0, ctx) = lower_binop op ctx in
      let (v1, ctx) = lower_expr e0 ctx in
      let (v2, ctx) = lower_expr e1 ctx in
      ctx |> Ctx.add_expr (Hir.ECall (v0, [v1; v2]))
  | Ast.ECall (e, es) ->
      lower_call e es ctx
  | Ast.EInvoke (e, x, es) ->
      let (v0, ctx) = lower_expr e ctx in
      let (v1, ctx) = Ctx.resolve_expr_path [x] [] ctx in
      let (vs, ctx) = es |> mapm lower_expr ctx in
      ctx |> Ctx.add_expr (Hir.ECall (v1, v0::vs))
  | Ast.ECast (e, t) ->
      let (v, ctx) = lower_expr e ctx in
      let (t, ctx) = lower_type t ctx in
      ctx |> Ctx.add_expr (Hir.ECast (v, t))
  | Ast.EIf (e, b0, b1) ->
      let (v, ctx) = lower_expr e ctx in
      let (b0, ctx) = lower_block b0 ctx in
      let (b1, ctx) = lower_block_opt b1 ctx in
      ctx |> Ctx.add_expr (Hir.EIf (v, b0, b1))
  | Ast.ELit l ->
      lower_lit l ctx
  | Ast.ELoop b ->
      let (b, ctx) = lower_block b ctx in
      ctx |> Ctx.add_expr (Hir.ELoop b)
  | Ast.ESelect (e0, e1) ->
      let (v0, ctx) = lower_expr e0 ctx in
      let (v1, ctx) = lower_expr e1 ctx in
      ctx |> Ctx.select_array v0 v1
  | Ast.ERecord (fs, _) ->
      let (fs, ctx) = fs |> mapm lower_field_expr ctx in
      ctx |> Ctx.add_expr (Hir.ERecord fs)
  | Ast.EUnOp (op, e) ->
      let (v0, ctx) = lower_unop op ctx in
      let (v1, ctx) = lower_expr e ctx in
      ctx |> Ctx.add_expr (Hir.ECall (v0, [v1]))
  | Ast.EReturn e ->
      let (v, ctx) = lower_expr_opt e ctx in
      ctx |> Ctx.add_expr (Hir.EReturn v)
  | Ast.EBreak e ->
      let (v, ctx) = lower_expr_opt e ctx in
      ctx |> Ctx.add_expr (Hir.EBreak v)
  | Ast.EContinue ->
      ctx |> Ctx.add_expr Hir.EContinue
  (* Desugared expressions *)
  | Ast.ETuple es ->
      let (vs, ctx) = es |> mapm lower_expr ctx in
      let fs = vs |> Hir.indexes_to_fields in
      ctx |> Ctx.add_expr (Hir.ERecord fs)
  | Ast.EProject (e, i) ->
      let (v, ctx) = lower_expr e ctx in
      ctx |> Ctx.add_expr (Hir.EAccess (v, Hir.index_to_field i))
  | Ast.EBlock b ->
      let ((ss, v), ctx) = lower_block b ctx in
      let ctx = ctx |> Ctx.add_stmts ss in
      (v, ctx)
  | Ast.EFunc (ps, e) ->
(*       Pretty_ast.pr_expr e Pretty.Ctx.brief; *)
(*       Ctx.print_scopes ctx.vstack Pretty.Ctx.brief; *)
      lower_closure ps e ctx
  | Ast.ETask (_ps, _b) ->
      todo ()
(*       let (x, ctx) = ctx |> Ctx.fresh_x in *)
(*       let (xs, ctx) = ctx |> Ctx.push_namespace x in *)
(*  *)
(*       let (ps, ctx) = ps |> mapm lower_param ctx in *)
(*       let ctx = ctx |> Ctx.push_vscope in *)
(*       let (v, ctx) = lower_block b ctx in *)
(*       let (ss0, ctx) = ctx |> Ctx.pop_vscope in *)
(*       let b = (ss0, v) in *)
(*  *)
(*       let ctx = ctx |> Ctx.pop_namespace in *)
(*       let ctx = ctx |> Ctx.add_item xs (Hir.ITask ([], [], [], ps, b)) in *)
(*  *)
(*       let (v, ctx) = ctx |> Ctx.add_expr (Hir.EItem (xs, [])) in *)
(*       ctx |> Ctx.add_expr (Hir.ECall (v, [])) *)
  | Ast.EFor (_p, e, b) ->
      let _p = todo () in
      let (_v, ctx) = lower_expr e ctx in
      let (_b, ctx) = lower_block b ctx in
      (todo (), ctx)
  | Ast.EWhile (e, b) ->
      let ctx = ctx |> Ctx.push_vscope in
      let (v0, ctx) = lower_expr e ctx in
      (* Then-branch *)
      let (b0, ctx) = lower_block b ctx in
      (* Else-branch *)
      let ctx = ctx |> Ctx.push_vscope in
      let (v1, ctx) = ctx |> Ctx.add_expr (Hir.ELit Ast.LUnit) in
      let (v1, ctx) = ctx |> Ctx.add_expr (Hir.EBreak v1) in
      let (ss1, ctx) = ctx |> Ctx.pop_vscope in
      let b1 = (ss1, v1) in
      (* If-stmt *)
      let (v2, ctx) = ctx |> Ctx.add_expr (Hir.EIf (v0, b0, b1)) in
      let (ss2, ctx) = ctx |> Ctx.pop_vscope in
      let b2 = (ss2, v2) in
      ctx |> Ctx.add_expr (Hir.ELoop b2)
  | Ast.EWhileVal _ ->
      todo ()
  | Ast.EIfVal (p, e, b0, b1) ->
      let (v, ctx) = lower_expr e ctx in
      let e0 = Ast.EBlock b0 in
      let e1 = match b1 with
      | Some b1 -> Ast.EBlock b1
      | None -> Ast.ELit Ast.LUnit
      in
      let c0 = ([(v, p)], [], e0) in
      let c1 = ([], [], e1) in
      lower_clauses [c0; c1] ctx
  | Ast.EMatch (e, arms) ->
      let (v, ctx) = lower_expr e ctx in
      let cs = Hir.arms_to_clauses arms v in
      lower_clauses cs ctx
  | Ast.EReceive e ->
      let (v, ctx) = lower_expr e ctx in
      ctx |> Ctx.add_expr (Hir.EReceive v)
  | Ast.EEmit (e0, e1) ->
      let (v0, ctx) = lower_expr e0 ctx in
      let (v1, ctx) = lower_expr e1 ctx in
      ctx |> Ctx.add_expr (Hir.EEmit (v0, v1))
  | Ast.EOn _ ->
      todo ()
(*       let ctx = ctx |> Ctx.push_vscope in *)
(*       let (rs, ctx) = rs |> mapm lower_receiver ctx in *)
(*       let (ss, ctx) = ctx |> Ctx.pop_vscope in *)
(*       let ctx = ctx |> Ctx.add_stmts (ss |> List.rev) in *)
(*       (v2, ctx) *)
  | Ast.EPath (xs, ts) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      ctx |> Ctx.resolve_expr_path xs ts
  | Ast.ECompr _ ->
      todo ()
  | Ast.EFrom _ ->
      todo ()

and lower_receiver (p, e0, e1) ctx =
  let (v0, ctx) = lower_expr e0 ctx in
  let ctx = ctx |> Ctx.push_vscope in
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let ctx = lower_irrefutable_pat p t v0 ctx in
  let (v1, ctx) = lower_expr e1 ctx in
  let (ss, ctx) = ctx |> Ctx.pop_vscope in
  (v0, (ss, v1), ctx)

and lower_unop op ctx =
  match op with
  | Ast.UNot -> ctx |> Ctx.add_expr (Hir.EItem (["not"], []))
  | Ast.UNeg -> ctx |> Ctx.add_expr (Hir.EItem (["neg"], []))

and lower_compr_clauses cs e0 ctx =
  match cs with
  | c::cs ->
      begin match c with
      | Ast.CFor (p, e1) ->
          let ctx = ctx |> Ctx.push_vscope in
          let (x, ctx) = lower_expr e1 ctx in
          let (t, ctx) = ctx |> Ctx.fresh_t in
          let ctx = lower_irrefutable_pat p t x ctx in
          let ((ss0, v), ctx) = lower_compr_clauses cs e0 ctx in
          let (ss1, ctx) = ctx |> Ctx.pop_vscope in
          let ss = ss0 @ ss1 in
          ((ss, v), ctx)
      | Ast.CIf e1 ->
          let (v, ctx) = lower_expr e1 ctx in
          let ctx = ctx |> Ctx.push_vscope in
          let ((ss0, v0), ctx) = lower_compr_clauses cs e0 ctx in
          let (ss1, ctx) = ctx |> Ctx.pop_vscope in
          let b0 = (ss0 @ ss1, v0) in
          let (b1, ctx) = lower_block_opt None ctx in
          let (v, ctx) = ctx |> Ctx.add_expr (Hir.EIf (v, b0, b1)) in
          (([], v), ctx)
      end
  | [] ->
      let (v, ctx) = lower_expr e0 ctx in
      (([], v), ctx)

and lower_field_expr (x, e) (ctx:Ctx.t) =
  match e with
  | Some e ->
    let (v, ctx) = lower_expr e ctx in
    ((x, v), ctx)
  | None ->
      match ctx |> Ctx.find_vname x with
      | Some (v, MVal) -> ((x, v), ctx)
      | Some (v, MVar) ->
          let (v, ctx) = Ctx.read_cell v ctx in
          ((x, v), ctx)
      | None -> panic "Name not found"

and lower_field_type (x, t) (ctx:Ctx.t) =
  match t with
  | Some t ->
    let (t, ctx) = lower_type t ctx in
    ((x, t), ctx)
  | None ->
    let (t, ctx) = ctx |> Ctx.fresh_t in
    ((x, t), ctx)

and lower_type_or_fresh t (ctx:Ctx.t) =
  match t with
  | Some t -> lower_type t ctx
  | None -> ctx |> Ctx.fresh_t

and unit_type = Hir.TNominal (["unit"], [])

and lower_type_or_unit t (ctx:Ctx.t) =
  match t with
  | Some t -> lower_type t ctx
  | None -> (unit_type, ctx)

and lower_type t (ctx:Ctx.t) =
  match t with
  | Ast.TFunc (ts, t) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = lower_type t ctx in
      (Hir.TFunc (ts, t), ctx)
  | Ast.TTuple ts ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let fs = ts |> Hir.indexes_to_fields in
      let t = fs |> fields_to_rows Hir.TRowEmpty in
      (Hir.TRecord t, ctx)
  | Ast.TRecord (fs, t) ->
      let (fs, ctx) = fs |> mapm lower_field_type ctx in
      let (t, ctx) = match t with
      | Some t -> lower_type t ctx
      | None -> (Hir.TRowEmpty, ctx)
      in
      let t = fs |> fields_to_rows t in
      (Hir.TRecord t, ctx)
  | Ast.TPath (xs, ts) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      ctx |> Ctx.resolve_type_path xs ts
  | Ast.TArray t ->
      let (t, ctx) = lower_type t ctx in
      (Hir.TNominal (["Array"], [t]), ctx)

(* Lowers an irrefutable pattern matching on variable v, e.g., val p = v; *)
and lower_irrefutable_pat p t v ctx =
  match p with
  | Ast.PIgnore -> ctx
  | Ast.POr _ -> panic "Found refutable pattern"
  | Ast.PRecord (pfs, _tail) ->
      pfs |> List.fold_left (fun ctx (x, p) ->
        let (v, ctx) = ctx |> Ctx.add_typed_expr (Hir.EAccess (v, x)) t in
        let (t, ctx) = ctx |> Ctx.fresh_t in
        match p with
        | Some p -> lower_irrefutable_pat p t v ctx
        | None ->
            let (_, ctx) = ctx |> Ctx.bind_vname x MVal in
            ctx
      ) ctx
  | Ast.PTuple ps ->
      let (_, ctx) = ps |> List.fold_left (fun (i, ctx) p ->
        let x = Hir.index_to_field i in
        let (v, ctx) = ctx |> Ctx.add_typed_expr (Hir.EAccess (v, x)) t in
        let (t, ctx) = ctx |> Ctx.fresh_t in
        (i+1, lower_irrefutable_pat p t v ctx)
      ) (0, ctx) in
      ctx
  | Ast.PConst _ -> panic "Found refutable pattern"
  | Ast.PVar x ->
      let (_, ctx) = ctx |> Ctx.bind_vname x MVal in
      ctx
  | Ast.PUnwrap _ -> panic "Found refutable pattern" (* TODO: Might be irrefutable *)

and branching_heuristic (eqs, _, _) cs =
  eqs
    |> List.map (fun (x, _) -> x)
    |> max_by (fun x -> cs
      |> List.filter (fun (eqs, _, _) -> eqs
        |> List.assoc_opt x
        |> Option.is_some))

and create_fresh_evs ctx l =
  let (l, ctx) = l |> List.fold_left (fun (l, ctx) _ -> 
      let (x, ctx) = ctx |> Ctx.fresh_x in
      (x::l, ctx)
    ) ([], ctx) in
  (l |> List.rev, ctx)

and compatible fs0 fs1 =
  let xs0 = fs0 |> List.map (fun (x, _) -> x) |> List.sort String.compare in
  let xs1 = fs1 |> List.map (fun (x, _) -> x) |> List.sort String.compare in
  List.combine xs0 xs1 |> List.for_all (fun (x0, x1) -> x0 = x1)

and fields_to_patterns fs =
  fs |> List.map (function
    | (_, Some p) -> p
    | (x, None) -> Ast.PVar x
  )

(* Unfold field accesses in the current block *)
and simplify_record (eqs, substs, ctx) v fs =
  fs |> List.fold_left (fun (eqs, substs, ctx) (x, p) ->
    match p with
    | Some p ->
        let (v, ctx) = ctx |> Ctx.add_expr (Hir.EAccess (v, x)) in
        simplify_eq (eqs, substs, ctx) (v, p)
    | None ->
        let (v, ctx) = ctx |> Ctx.add_expr (Hir.EAccess (v, x)) in
        simplify_eq (eqs, substs, ctx) (v, Ast.PVar x)
  ) (eqs, substs, ctx)

and simplify_tuple (eqs, substs, ctx) v ps =
  let fs = ps |> List.map (fun p -> Some p) |> Hir.indexes_to_fields in
  simplify_eq (eqs, substs, ctx) (v, Ast.PRecord (fs, None))

(* and simplify_or (eqs, substs, ctx) cs p0 p1 = *)
(*   let (v0, ctx) = ctx |> Ctx.fresh_x in *)
(*   let (v1, ctx) = ctx |> Ctx.fresh_x in *)
(*   let eqs0 = (v0, p0)::eqs in *)
(*   let eqs1 = (v1, p1)::eqs in *)
(*   let cs = (eqs0, substs, e)::(eqs1, substs, e)::(cs |> List.tl) in *)
(*   lower_clauses cs ctx *)

(* Simplifies an equation such that it only contains refutable patterns. *)
and simplify_eq ((eqs, substs, ctx) as acc) (v, p) =
  match p with
  | Ast.PVar x -> (eqs, (x, v)::substs, ctx) (* Substitute *)
  | Ast.PIgnore -> acc (* Ignore completely *)
  | Ast.PRecord (fs, _tail) -> simplify_record acc v fs
  | Ast.PTuple ps -> simplify_tuple acc v ps
  | Ast.POr (_p0, _p1) -> todo ()
  (* Refutable patterns are not simplified *)
  | Ast.PConst _ | Ast.PUnwrap _ -> ((v, p)::eqs, substs, ctx)

and simplify_clause (eqs, substs, e) ctx =
  let (eqs', substs', ctx) = eqs |> List.fold_left simplify_eq ([], [], ctx) in
  let eqs' = eqs' |> List.rev in
  let substs' = substs' |> List.rev |> List.append substs in
  ((eqs', substs', e), ctx)

and simplify_clauses cs ctx =
  cs |> mapm simplify_clause ctx

(* Returns true if xs0 and xs1 are variants of the same enum *)
and same_enum xs0 xs1 =
  match xs0, xs1 with
  | [_], [_] -> true
  | h0::t0, h1::t1 ->
      if h0 = h1 then
        same_enum t0 t1
      else
        false
  | _ -> false

and branch_variant branching_v xs0 cs ctx : (var * Ctx.t) =
  let (xs0, decl) = ctx |> Ctx.resolve_path xs0 in
  let (ts0, ctx) = fresh_variant_ts decl ctx in
  (* Create fresh variable for the inner pattern *)
  let (v, ctx) = ctx |> Ctx.fresh_x in
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let ctx = cs |> List.fold_left (fun ctx ((eqs, substs, e) as c) ->
    match eqs |> List.assoc_opt branching_v with
    | Some Ast.PUnwrap (xs1, ps1) ->
        let (xs1, _) = ctx |> Ctx.resolve_path xs1 in
        if xs0 = xs1 then
          (* Push clauses with equivalent pattern matches on the branching
            * variable to the then-branch *)
          let eqs = eqs |> List.remove_assoc branching_v in
          let p = match ps1 with
          | [] -> todo ()
          | [p] -> p
          | ps -> Ast.PTuple ps
          in
          let eqs = (v, p)::eqs in
          ctx |> Ctx.add_then_clause (eqs, substs, e)
        else
          if same_enum xs0 xs1 then
            (* Push clauses with other pattern matches on the branching
            * variable to the else-branch *)
            ctx |> Ctx.add_else_clause c
          else
            panic "Branching on different enums"
    | None ->
        (* Clauses which do not match on the branching variable are be
          * pushed to both branches *)
        ctx |> Ctx.add_then_clause c
            |> Ctx.add_else_clause c
    | _ -> panic "Branching on different patterns"
  ) ctx in
  let (then_cs, else_cs, ctx) = ctx |> Ctx.take_clauses in

  (* Create then-branch *)
  let ctx = ctx |> Ctx.push_vscope in
  let (then_v, ctx) = lower_clauses then_cs ctx in
  let (then_ss, ctx) = ctx |> Ctx.pop_vscope in
  let then_b = ((v, t, EUnwrap (xs0, ts0, branching_v))::then_ss, then_v) in

  (* Create else-branch *)
  let ctx = ctx |> Ctx.push_vscope in
  let (else_v, ctx) = lower_clauses else_cs ctx in
  let (else_ss, ctx) = ctx |> Ctx.pop_vscope in
  let else_b = (else_ss, else_v) in

  let (v, ctx) = ctx |> Ctx.add_expr (Hir.EIs (xs0, ts0, branching_v)) in
  let (v, ctx) = ctx |> Ctx.add_expr (Hir.EIf (v, then_b, else_b)) in
  (v, ctx)

and fresh_variant_ts decl ctx =
  match decl with
  | Table.DVariant n -> Ctx.fresh_ts n ctx
  | _ -> unreachable ()

(* Branch clauses on a head clause c *)
and branch_clauses ((eqs, _, _) as c) cs ctx =
  (* Get the variable in the first clause which occurs in the most equations *)
  let branching_v = branching_heuristic c cs in
  (* Find the pattern *)
  match eqs |> List.assoc branching_v with
  (* Irrefutable top-level patterns are eliminated earlier through simplify_clauses *)
  | Ast.PVar _ | Ast.PIgnore | Ast.PRecord _ | Ast.PTuple _ | Ast.POr _ -> unreachable ()
  | Ast.PConst _l -> todo ()
  | Ast.PUnwrap (xs, _) -> branch_variant branching_v xs cs ctx

and lower_clauses cs ctx : (var * Ctx.t) =
  if cs = [] then panic "Non-exhaustive match";
  let (cs, ctx) = simplify_clauses cs ctx in
  (* First clause has precedence over others *)
  let (eqs, substs, e) as c = cs |> List.hd in
  if eqs = [] then
    (* This pattern equation is now solved *)
    let ctx = substs |> List.fold_left (fun ctx (x, v) -> ctx |> Ctx.rename_vname x (v, MVal)) ctx in
    (* TODO: Duplication leads to duplicated lowerings, we might need sharing. *)
    lower_expr e ctx
  else
    branch_clauses c cs ctx

and lower_stmt s (ctx:Ctx.t) =
  match s with
  | Ast.SNoop -> ctx
  | Ast.SVal ((p, t), e) ->
      let (v, ctx) = lower_expr e ctx in
      let (t, ctx) = lower_type_or_fresh t ctx in
      (* One of the few places where we need to type-annotate *)
      let ctx = ctx |> Ctx.annotate v t in
      begin match p with
      | Ast.PVar x -> ctx |> Ctx.rename_vname x (v, MVal)
      | _ -> lower_irrefutable_pat p t v ctx
      end
  | Ast.SVar ((x, t), e) ->
      let (v, ctx) = lower_expr e ctx in
      let (t, ctx) = lower_type_or_fresh t ctx in
      let (v, ctx) = ctx |> Ctx.new_cell v t in
      ctx |> Ctx.rename_vname x (v, MVar)
  | Ast.SExpr e -> 
      let (_, ctx) = lower_expr e ctx in
      ctx

and lower_block_opt b (ctx:Ctx.t) =
  match b with
  | Some b ->
      lower_block b ctx
  | None -> 
      ctx |> Ctx.empty_block

and lower_block (ss, e) (ctx:Ctx.t) =
  let ctx = ctx |> Ctx.push_vscope in
  let ctx = ss |> List.fold_left (fun ctx s -> lower_stmt s ctx) ctx in
  let (v, ctx) = lower_expr_opt e ctx in
  let (ss, ctx) = ctx |> Ctx.pop_vscope in
  ((ss, v), ctx)

and lower_binop op (ctx:Ctx.t) =
  let x = Ast.binop_name op in
  ctx |> Ctx.add_expr (Hir.EItem ([x], []))

and splice_regex = (Str.regexp "\\${[^}]+}\\|\\$[a-zA-Z_][a-zA-Z0-9_]*")

and lower_lit l (ctx:Ctx.t) =
  match l with
  (* Lower interpolated string literals *)
  | Ast.LString c ->
      let (vs, ctx) = c |> Str.full_split splice_regex
        |> mapm (fun s ctx ->
          match s with
          | Str.Text s ->
              ctx |> Ctx.add_expr (Hir.ELit (Ast.LString s))
          | Str.Delim s ->
              let s = String.sub s 1 ((String.length s) - 1) in
              let e = Parser.expr Lexer.main (Lexing.from_string s) in
              let ctx = ctx |> Ctx.push_vscope in
              let (v0, ctx) = lower_expr e ctx in
              let (ss, ctx) = ctx |> Ctx.pop_vscope in
              let ctx = ctx |> Ctx.add_stmts ss in
              let (v1, ctx) = ctx |> Ctx.add_expr (Hir.EItem (["to_string"], [])) in
              ctx |> Ctx.add_expr (Hir.ECall (v1, [v0]))
        ) ctx in
      begin match vs with
      | v::vs ->
        vs |> List.fold_left (fun (v1, ctx) v2 ->
          let (v0, ctx) = ctx |> Ctx.add_expr (Hir.EItem (["concat"], [])) in
          ctx |> Ctx.add_expr (Hir.ECall (v0, [v1; v2]))
        ) (v, ctx)
      | _ -> unreachable ()
      end
  | _ -> ctx |> Ctx.add_expr (Hir.ELit l)

and lower_closure ps e (ctx:Ctx.t) =

  (* Compile the block body *)
  let ctx = ctx |> Ctx.push_vscope in
  let (ps, ctx) = ps |> mapm lower_param ctx in
  let ((ss0, v0), ctx) = lower_block e ctx in
  let (ss1, ctx) = ctx |> Ctx.pop_vscope in

  let fvs = Hir.free_vars ps (ss0, v0) |> Hir.indexes_to_fields in

  (* Create an extra function parameter for the environment *)
  let (v, ctx) = ctx |> Ctx.fresh_x in
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let p = (v, t) in
  let ps = ps @ [p] in

  (* Create code for unpacking the closure record and its into parameters inside the function *)
  let (ss2, ctx) = lambda_unpack v fvs ctx in
  let b1 = (ss2 @ ss1 @ ss0, v0) in

  (* Create the function *)
  let (xs, ctx) = ctx |> Ctx.fresh_f in
  let (t, ctx) = ctx |> Ctx.fresh_t in
  let ctx = ctx |> Ctx.add_item xs (Hir.IDef ([], [], ps, t, b1)) in

  (* Create the function pointer *)
  let (v, ctx) = ctx |> Ctx.add_expr (Hir.EItem (xs, [])) in

  (* Create code for packing the function-pointer + free variables into a record outside the function *)
  lambda_pack v fvs ctx

(* Call a closure by getting the function pointer and passing in the record as an extra parameter *)
and lambda_call v0 vs ctx =
  let (v1, ctx) = ctx |> Ctx.add_expr (Hir.EAccess (v0, "f")) in
  let (v2, ctx) = ctx |> Ctx.add_expr (Hir.EAccess (v0, "r")) in
  let (v3, ctx) = ctx |> Ctx.add_expr (Hir.ECall (v1, vs @ [v2])) in
  (v3, ctx)

(* Lift a closure into a top-level function. *)
(* Need to lift all variables inside the closure which are defined outside the closure *)
and lambda_pack v0 fvs ctx =
  let (v1, ctx) = ctx |> Ctx.add_expr (Hir.ERecord fvs) in
  let (v2, ctx) = ctx |> Ctx.add_expr (Hir.ERecord [("f", v0); ("r", v1)]) in
  (v2, ctx)

and lambda_unpack v0 fvs ctx =
  let ctx = ctx |> Ctx.push_vscope in
  let ctx = fvs |> foldl (fun ctx (x, v1) ->
      ctx |> Ctx.add_named_expr (Hir.EAccess (v0, x)) v1 |> snd
  ) ctx in
  ctx |> Ctx.pop_vscope
