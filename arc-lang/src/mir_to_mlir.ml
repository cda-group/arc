open Utils

module Ctx = struct
  type t = {
    mir: Mir.mir;
    mlir: Mlir.mlir;
    stack: scope list;
    subctx: subctx list;
  }
  and scope = {
    vsubst: (Mlir.name * Mlir.ty) list;
  }
  and subctx =
    | STask of {
      input_handle: Mlir.arg;
      output_handle: Mlir.arg
    }

  let add_item xs i ctx = { ctx with mlir = (xs, i)::ctx.mlir }
  let make mir = { mir; mlir = []; stack = []; subctx = []; }

  let push_subctx c ctx = { ctx with subctx = c::ctx.subctx }
  let pop_subctx ctx = { ctx with subctx = ctx.subctx |> tl }

  let push_scope ctx = { ctx with stack = { vsubst = [] }::ctx.stack }
  let pop_scope ctx = { ctx with stack = ctx.stack |> tl }

  let get_input_handle ctx = match ctx.subctx |> hd with
  | STask {input_handle; _} -> input_handle

  let get_output_handle ctx = match ctx.subctx |> hd with
  | STask {output_handle; _} -> output_handle

  let bind_var x (t:Mlir.ty) ctx =
    match ctx.stack with
    | hd::tl ->
        { ctx with stack = { vsubst = (x, t)::hd.vsubst}::tl }
    | _ -> unreachable ()

  let typeof_opt x ctx =
    ctx.stack |> List.find_map (fun scope -> scope.vsubst |> List.assoc_opt x)

  let typeof x ctx = ctx |> typeof_opt x |> Option.get

end

let id x = x

let rec mlir_of_thir mir =
  let ctx = Ctx.make mir in
  let ctx = mir |> List.fold_left codegen_item ctx in
  ctx.mlir

and filter_unit ts =
  ts |> filter (fun t -> t |> Hir.is_unit |> not)

and codegen_item ctx ((xs, ts), i) =
  match i with
  | Hir.IVal (t0, b) ->
      let (x, ctx) = codegen_path (xs, ts) ctx in
      let (t0, ctx) = codegen_type t0 ctx in
      let terminator _ = Mlir.ENoop in
      let (b, ctx) = codegen_block b terminator ctx in
      ctx |> Ctx.add_item x (Mlir.IAssign (t0, b))
  | Hir.IEnum _ -> ctx
  | Hir.IExternDef (_, ts, t) ->
      let (x, ctx) = codegen_path (xs, ts) ctx in
      let (ts, ctx) = ts |> filter_unit |> mapm codegen_type ctx in
      let ps = ts |> Hir.indexes_to_fields in
      let (t, ctx) = codegen_type t ctx in
      ctx |> Ctx.add_item x (Mlir.IExternFunc (ps, t))
  | Hir.IExternType _ -> ctx
  | Hir.IDef (_, ps, t, b) ->
      let ctx = ctx |> Ctx.push_scope in
      let (x, ctx) = codegen_path (xs, ts) ctx in
      let (ps, ctx) = ps |> mapm_filter codegen_param ctx in
      let terminator (v, t) = Mlir.EReturn (t |> Option.map (fun t -> (v, t))) in
      let (t, ctx) = codegen_type t ctx in
      let (b, ctx) = codegen_block b terminator ctx in
      let ctx = ctx |> Ctx.pop_scope in
      ctx |> Ctx.add_item x (Mlir.IFunc (ps, t, b))
  | Hir.ITask (_, ps, (xs0, _), (xs1, _), b) ->
      let ctx = ctx |> Ctx.push_scope in
      let (x, ctx) = codegen_path (xs, ts) ctx in
      let (ps, ctx) = ps |> mapm_filter codegen_param ctx in
      let (t0, ctx) = codegen_enum (xs0, ts) ctx in
      let (t1, ctx) = codegen_enum (xs1, ts) ctx in
      let param0 = ("in", t0) in
      let param1 = ("out", Mlir.TStream t1) in
      let ps = ps @ [param0; param1] in
      let ctx = ctx |> Ctx.push_subctx (Ctx.STask {input_handle=param0; output_handle=param1})  in
      let (b, ctx) = codegen_block b (fun _ -> Mlir.EReturn None) ctx in
      let ctx = ctx |> Ctx.pop_subctx in
      ctx |> Ctx.add_item x (Mlir.ITask (ps, b))
  | Hir.ITypeAlias _ -> ctx
  | Hir.IVariant _ -> ctx
  | Hir.IClass _ -> ctx
  | Hir.IInstance _ -> ctx
  | Hir.IInstanceDef _ -> ctx
  | Hir.IClassDecl _ -> ctx

and codegen_enum (xs, ts) (ctx:Ctx.t) =
  match ctx.mir |> List.assoc (xs, ts) with
  | Hir.IEnum (_, xss) ->
      let (vs, ctx) = xss |> mapm (fun xs ctx -> codegen_variant (xs, ts) ctx) ctx in
      (Mlir.TEnum vs, ctx)
  | _ -> unreachable ()

and codegen_extern_param (t:Hir.ty) (ctx:Ctx.t) =
  if Hir.is_unit t then
    (None, ctx)
  else
    let (t, ctx) = codegen_type t ctx in
    (Some t, ctx)

and codegen_param ((x, t):Hir.param) (ctx:Ctx.t) =
  if Hir.is_unit t then
    (None, ctx)
  else
    let (t, ctx) = codegen_type t ctx in
    let ctx = ctx |> Ctx.bind_var x t in
    (Some (x, t), ctx)

and nominal xs = Hir.TNominal (xs, [])
and atom x = Hir.TNominal ([x], [])

and ts_of_vs vs = vs |> List.map (fun (_, t) -> t)
and fts_of_fvs vs = vs |> List.map (fun (x, (_, t)) -> (x, t))

and codegen_row r ctx =
  let rec codegen_row r ctx acc =
    match r with
    | Hir.TRowExtend ((x, t), r) ->
        let (t, ctx) = codegen_type t ctx in
        codegen_row r ctx ((x, t)::acc)
    | Hir.TRowEmpty -> (acc, ctx)
    | _ -> unreachable ()
  in
  let (fts, ctx) = codegen_row r ctx [] in
  (fts |> List.rev, ctx)

and codegen_type t ctx =
  match t with
  | Hir.TFunc (ts, t) ->
      let (ts, ctx) = ts |> mapm codegen_type ctx in
      let (t, ctx) = codegen_type t ctx in
      (Mlir.TFunc (ts, t), ctx)
  | Hir.TRecord r ->
      let (fts, ctx) = codegen_row r ctx in
      (Mlir.TRecord fts, ctx)
  | Hir.TNominal (xs, ts) ->
      (* Intrinsic types in MLIR *)
      begin match xs with
      | ["bool"] -> (Mlir.TNative "i1", ctx)
      | ["i16"] -> (Mlir.TNative "i16", ctx)
      | ["i32"] -> (Mlir.TNative "i32", ctx)
      | ["i64"] -> (Mlir.TNative "i64", ctx)
      | ["i128"] -> (Mlir.TNative "i128", ctx)
      | ["f32"] -> (Mlir.TNative "f32", ctx)
      | ["f64"] -> (Mlir.TNative "f64", ctx)
      | ["unit"] -> (Mlir.TNative "()", ctx)
      | _ ->
          begin match ctx.mir |> List.assoc (xs, ts) with
          | Hir.IEnum (_, xss) ->
              let (vs, ctx) = xss |> mapm (fun xs ctx -> codegen_variant (xs, ts) ctx) ctx in
              (Mlir.TEnum vs, ctx)
          | Hir.IExternType _ ->
              let (xs, ctx) = codegen_path (xs, ts) ctx in
              (Mlir.TAdt (xs, []), ctx)
          | _ -> unreachable ()
          end
      end
  | Hir.TVar _
  | Hir.TGeneric _
  | Hir.TRowEmpty
  | Hir.TRowExtend _ -> unreachable ()

and codegen_block (ss, v) terminator ctx =
  let ctx = ctx |> Ctx.push_scope in
  let (ss, ctx) = ss |> mapm codegen_ssa ctx in
  let v = terminator (v, ctx |> Ctx.typeof_opt v) in
  let ctx = ctx |> Ctx.pop_scope in
  (ss @ [(None, v)], ctx)

and codegen_path ((xs, _ts):(Hir.path * Hir.ty list)) ctx =
  (xs |> String.concat "__", ctx)

and codegen_ssa (v, t, e) ctx =
  let (v, ctx) = codegen_param (v, t) ctx in
  let (e, ctx) = codegen_expr t e ctx in
  ((v, e), ctx)

and codegen_expr t e ctx =
  let typeof v = ctx |> Ctx.typeof v in
  let arg v = (v, typeof v) in
  let arg_opt v = ctx |> Ctx.typeof_opt v |> Option.map (fun t -> (v, t)) in
  let args vs = vs |> List.filter_map arg_opt in
  match e with
  | Hir.EAccess (v0, x0) ->
      (Mlir.EAccess (arg v0, x0), ctx)
  | Hir.EAfter _ -> todo ()
  | Hir.EEvery _ -> todo ()
  | Hir.EEq (v0, v1) ->
      (Mlir.EBinOp (Mlir.BEqu (int_or_float_or_bool t), arg v0, arg v1), ctx)
  | Hir.ECall (v0, vs) ->
      (Mlir.ECall (arg v0, args vs), ctx)
  | Hir.ECast _ -> todo ()
  | Hir.EEmit v0 ->
      (Mlir.EEmit (ctx |> Ctx.get_output_handle, (arg v0)), ctx)
  | Hir.EEnwrap (xs, ts, v0) ->
      let (x, ctx) = codegen_path (xs, ts) ctx in
      (Mlir.EEnwrap (x, arg_opt v0), ctx)
  | Hir.EUnwrap (xs, ts, v0) ->
      if Hir.is_unit t then
        (Mlir.ENoop, ctx)
      else
        let (x, ctx) = codegen_path (xs, ts) ctx in
        (Mlir.EUnwrap (x, arg v0), ctx)
  | Hir.EIs (xs, ts, v0) ->
      let (x, ctx) = codegen_path (xs, ts) ctx in
      (Mlir.EIs (x, arg v0), ctx)
  | Hir.EIf (v0, b0, b1) ->
      let terminator (v, t) = Mlir.EResult (t |> Option.map (fun t -> (v, t))) in
      let (b0, ctx) = codegen_block b0 terminator ctx in
      let (b1, ctx) = codegen_block b1 terminator ctx in
      (Mlir.EIf (arg v0, b0, b1), ctx)
  | Hir.ELit (Ast.LUnit) ->
      (Mlir.ENoop, ctx)
  | Hir.ELit l ->
      let (l, ctx) = codegen_lit l ctx in
      (Mlir.EConst l, ctx)
  | Hir.ELoop b ->
      let terminator _ = Mlir.EYield in
      let (b, ctx) = codegen_block b terminator ctx in
      (Mlir.ELoop b, ctx)
  | Hir.EReceive ->
      (Mlir.EReceive (ctx |> Ctx.get_input_handle), ctx)
  | Hir.ERecord fvs ->
      let (fvs, ctx) = fvs |> mapm_filter codegen_field_expr ctx in
      (Mlir.ERecord fvs, ctx)
  | Hir.EReturn v0 ->
      (Mlir.EReturn (arg_opt v0), ctx)
  | Hir.EBreak v0 ->
      (Mlir.EBreak (arg_opt v0), ctx)
  | Hir.EContinue ->
      (Mlir.EContinue, ctx)
  | Hir.EItem (xs, ts) ->
      match ctx.mir |> List.assoc (xs, ts) with
      | Hir.IVal _ -> todo ()
      | Hir.IEnum _ -> unreachable ()
      | Hir.IExternDef (xs, _, _) ->
          let (x, ctx) = codegen_path (xs, ts) ctx in
          (Mlir.EConst (Mlir.CFun x), ctx)
      | Hir.IExternType _ -> unreachable ()
      | Hir.IDef _ ->
          let (x, ctx) = codegen_path (xs, ts) ctx in
          (Mlir.EConst (Mlir.CFun x), ctx)
      | Hir.ITask _ -> todo ()
      | Hir.ITypeAlias (_ps, _t1) -> unreachable ()
      | Hir.IVariant _ -> unreachable ()
      | Hir.IClass _ -> unreachable ()
      | Hir.IClassDecl _ -> unreachable ()
      | Hir.IInstance _ -> unreachable ()
      | Hir.IInstanceDef _ -> unreachable ()

and codegen_field_type (x, t) ctx =
  let (t, ctx) = codegen_type t ctx in
  ((x, t), ctx)

and codegen_field_expr (x, v) ctx =
  let fa = ctx |> Ctx.typeof_opt v |> Option.map (fun t -> (x, (v, t))) in
  (fa, ctx)

and codegen_variant ((xs, ts):(Hir.path * Hir.ty list)) ctx =
  match ctx.mir |> List.assoc (xs, ts) with
  | Hir.IVariant t ->
      let (t, ctx) = codegen_type t ctx in
      let (xs, ctx) = codegen_path (xs, ts) ctx in
      ((xs, t), ctx)
  | _ -> unreachable ()

and int_or_float t =
  match t with
  | t when Hir.is_int t -> Mlir.NInt
  | t when Hir.is_float t -> Mlir.NFlt
  | _ -> todo ()

and int_or_float_or_bool t =
  match t with
  | t when Hir.is_int t -> Mlir.EqInt
  | t when Hir.is_float t -> Mlir.EqFlt
  | t when Hir.is_bool t -> Mlir.EqBool
  | _ -> todo ()

and codegen_lit l ctx =
  let l = match l with
  | Ast.LInt (d, _) -> Mlir.CInt d
  | Ast.LFloat (f, _) -> Mlir.CFloat f
  | Ast.LBool b -> Mlir.CBool b
  | Ast.LString _ -> todo ()
  | Ast.LUnit -> todo ()
  | Ast.LChar _ -> todo ()
  in (l, ctx)
