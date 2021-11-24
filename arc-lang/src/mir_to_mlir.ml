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

let rec mlir_of_mir mir =
  let ctx = Ctx.make mir in
  let ctx = mir |> foldl (fun ctx a -> lower_item a ctx) ctx in
  ctx.mlir

and none_if_unit t ctx =
  if Mir.is_unit t then
    (None, ctx)
  else
    let (t, ctx) = lower_type t ctx in
    (Some t, ctx)

and filter_unit ts =
  ts |> filter (fun t -> t |> Mir.is_unit |> not)

and lower_item ((xs, ts), i) (ctx:Ctx.t) =
  match i with
  | Mir.IVal (_, t0, b) ->
      let (x, ctx) = lower_path (xs, ts) ctx in
      let (t0, ctx) = lower_type t0 ctx in
      let terminator _ = Mlir.ENoop in
      let (b, ctx) = lower_block b terminator ctx in
      ctx |> Ctx.add_item x (Mlir.IAssign (t0, b))
  | Mir.IEnum _ -> ctx
  | Mir.IExternDef (a, ps, t) ->
      if not (is_intrinsic a) then
        let (x, ctx) = ctx |> resolve_intrinsic a (xs, ts) in
        let (ts, ctx) = ps |> filter_unit |> mapm lower_type ctx in
        let ps = ts |> Hir.indexes_to_fields in
        let (t, ctx) = lower_type t ctx in
        ctx |> Ctx.add_item x (Mlir.IExternFunc (ps, t))
      else
        ctx
  | Mir.IExternType _ -> ctx
  | Mir.IDef (_, ps, t, b) ->
      let ctx = ctx |> Ctx.push_scope in
      let (x, ctx) = lower_path (xs, ts) ctx in
      let (ps, ctx) = ps |> mapm_filter lower_param ctx in
      let terminator (v, t) = Mlir.EReturn (t |> Option.map (fun t -> (v, t))) in
      let (t, ctx) = none_if_unit t ctx in
      let (b, ctx) = lower_block b terminator ctx in
      let ctx = ctx |> Ctx.pop_scope in
      ctx |> Ctx.add_item x (Mlir.IFunc (ps, t, b))
  | Mir.ITask (_, ps0, ps1, b) ->
      let ctx = ctx |> Ctx.push_scope in
      let (x, ctx) = lower_path (xs, ts) ctx in
      let (ps0, ctx) = ps0 |> mapm_filter lower_param ctx in
      let (ps1, ctx) = ps1 |> mapm_filter lower_param ctx in
      let (b, ctx) = lower_block b (fun _ -> Mlir.EReturn None) ctx in
      let ctx = ctx |> Ctx.pop_subctx in
      ctx |> Ctx.add_item x (Mlir.ITask (ps0, ps1, b))
  | Mir.IVariant _ -> ctx

and lower_enum (xs, ts) (ctx:Ctx.t) =
  match ctx.mir |> List.assoc (xs, ts) with
  | Mir.IEnum (_, xss) ->
      let (vs, ctx) = xss |> mapm (fun xs ctx -> lower_variant (xs, ts) ctx) ctx in
      (Mlir.TEnum vs, ctx)
  | _ -> unreachable ()

and lower_extern_param (t:Mir.ty) (ctx:Ctx.t) =
  if Mir.is_unit t then
    (None, ctx)
  else
    let (t, ctx) = lower_type t ctx in
    (Some t, ctx)

and lower_param ((x, t):Mir.param) (ctx:Ctx.t) =
  if Mir.is_unit t then
    (None, ctx)
  else
    let (t, ctx) = lower_type t ctx in
    let ctx = ctx |> Ctx.bind_var x t in
    (Some (x, t), ctx)

and lower_types ts ctx =
  ts |> mapm lower_type ctx

and lower_type t ctx =
  match t with
  | Mir.TFunc (ts, t) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = lower_type t ctx in
      (Mlir.TFunc (ts, t), ctx)
  | Mir.TRecord fts ->
      let (fts, ctx) = lower_type_fields fts ctx in
      (Mlir.TRecord fts, ctx)
  | Mir.TNominal (xs, ts) ->
      begin match ctx.mir |> List.assoc (xs, ts) with
      | Mir.IEnum (_, xss) ->
          let (vs, ctx) = xss |> mapm (fun xs ctx -> lower_variant (xs, ts) ctx) ctx in
          (Mlir.TEnum vs, ctx)
      | Mir.IExternType a ->
          begin
            match a |> List.assoc_opt "intrinsic" with
            | Some Some Ast.LString x ->
                (Mlir.TNative x, ctx)
            | None ->
              let (xs, ctx) = lower_path (xs, ts) ctx in
              (Mlir.TAdt (xs, []), ctx)
            | _ -> panic "Expected literal string, got something else"
          end
      | _ -> unreachable ()
      end

and lower_type_fields fts ctx =
  fts |> mapm lower_type_field ctx

and lower_type_field (x, t) ctx =
  let (t, ctx) = lower_type t ctx in
  ((x, t), ctx)

and lower_block (ss, v) terminator ctx =
  let ctx = ctx |> Ctx.push_scope in
  let (ss, ctx) = ss |> mapm lower_ssa ctx in
  let v = terminator (v, ctx |> Ctx.typeof_opt v) in
  let ctx = ctx |> Ctx.pop_scope in
  (ss @ [(None, v)], ctx)

and lower_path (xs, ts) ctx =
  (* Mangle path *)
  let (ts, ctx) = lower_types ts ctx in
  let s0 = xs |> String.concat "" in
  let s1 = mangle_types ts in
  (s0 ^ s1, ctx)

and mangle_types ts =
  ts |> map mangle_type |> String.concat ""

and mangle_type t =
  let rec mangle_type t acc =
    match t with
    | Mlir.TAdt (x, ts) ->
        let acc = "Adt"::acc in
        let acc = x::acc in
        let acc = mangle_types ts acc in
        let acc = "End"::acc in
        acc
    | Mlir.TEnum vts ->
        let acc = "Enum"::acc in
        let acc = mangle_fields vts acc in
        let acc = "End"::acc in
        acc
    | Mlir.TFunc (ts, t) ->
        let acc = "Func"::acc in
        let acc = mangle_types ts acc in
        let acc = mangle_type t acc in
        let acc = "End"::acc in
        acc
    | Mlir.TNative x ->
        x::acc
    | Mlir.TRecord fts ->
        let acc = "Struct"::acc in
        let acc = mangle_fields fts acc in
        let acc = "End"::acc in
        acc
    | Mlir.TStream t ->
        let acc = "Stream"::acc in
        let acc = mangle_type t acc in
        let acc = "End"::acc in
        acc
  and mangle_types ts acc =
    ts |> foldl (fun acc t -> mangle_type t acc) acc
  and mangle_field (x, t) acc =
    let acc = ((x |> String.length) |> Int.to_string)::acc in
    let acc = x::acc in
    let acc = mangle_type t acc in
    acc
  and mangle_fields fts acc =
    fts |> foldl (fun acc ft -> mangle_field ft acc) acc
  in
  mangle_type t [] |> List.rev |> String.concat ""

and lower_ssa (v, t, e) ctx =
  let (v, ctx) = lower_param (v, t) ctx in
  let (e, ctx) = lower_expr t e ctx in
  ((v, e), ctx)

and lower_expr t e ctx =
  let typeof v = ctx |> Ctx.typeof v in
  let arg v = (v, typeof v) in
  let arg_opt v = ctx |> Ctx.typeof_opt v |> Option.map (fun t -> (v, t)) in
  let args vs = vs |> List.filter_map arg_opt in
  match e with
  | Mir.EAccess (v0, x0) ->
      (Mlir.EAccess (arg v0, x0), ctx)
  | Mir.ECall (v0, vs) ->
      (Mlir.ECall (arg v0, args vs), ctx)
  | Mir.ECast _ -> todo ()
  | Mir.EEmit (v0, v1) ->
      (Mlir.EEmit ((arg v0), (arg v1)), ctx)
  | Mir.EEnwrap (xs, ts, v0) ->
      let (x, ctx) = lower_path (xs, ts) ctx in
      (Mlir.EEnwrap (x, arg_opt v0), ctx)
  | Mir.EUnwrap (xs, ts, v0) ->
      if Mir.is_unit t then
        (Mlir.ENoop, ctx)
      else
        let (x, ctx) = lower_path (xs, ts) ctx in
        (Mlir.EUnwrap (x, arg v0), ctx)
  | Mir.EIs (xs, ts, v0) ->
      let (x, ctx) = lower_path (xs, ts) ctx in
      (Mlir.EIs (x, arg v0), ctx)
  | Mir.EIf (v0, b0, b1) ->
      let terminator (v, t) = Mlir.EResult (t |> Option.map (fun t -> (v, t))) in
      let (b0, ctx) = lower_block b0 terminator ctx in
      let (b1, ctx) = lower_block b1 terminator ctx in
      (Mlir.EIf (arg v0, b0, b1), ctx)
  | Mir.ELit l ->
      let e = match l with
        | Ast.LInt (d, _) -> Mlir.EConst (Mlir.CInt d)
        | Ast.LFloat (f, _) -> Mlir.EConst (Mlir.CFloat f)
        | Ast.LBool b -> Mlir.EConst (Mlir.CBool b)
        | Ast.LString s -> Mlir.EConst (Mlir.CAdt (Printf.sprintf "String::from(\\\"%s\\\")" s))
        | Ast.LUnit -> Mlir.ENoop
        | Ast.LChar _ -> todo ()
      in
      (e, ctx)
  | Mir.ELoop b ->
      let terminator _ = Mlir.EYield in
      let (b, ctx) = lower_block b terminator ctx in
      (Mlir.ELoop b, ctx)
  | Mir.EReceive v ->
      (Mlir.EReceive (arg v), ctx)
  | Mir.ERecord fvs ->
      let (fvs, ctx) = fvs |> mapm_filter lower_field_expr ctx in
      let (xs, ts) = fvs |> map (fun (_, v) -> v) |> Utils.unzip in
      (Mlir.ERecord (xs, ts), ctx)
  | Mir.EReturn v0 ->
      (Mlir.EReturn (arg_opt v0), ctx)
  | Mir.EBreak v0 ->
      (Mlir.EBreak (arg_opt v0), ctx)
  | Mir.EContinue ->
      (Mlir.EContinue, ctx)
  | Mir.EItem (xs, ts) ->
      match ctx.mir |> List.assoc (xs, ts) with
      | Mir.IVal _ -> todo ()
      | Mir.IEnum _ -> unreachable ()
      | Mir.IExternDef (a, _, _) ->
          let (x, ctx) = ctx |> resolve_intrinsic a (xs, ts) in
          (Mlir.EConst (Mlir.CFun x), ctx)
      | Mir.IExternType _ -> unreachable ()
      | Mir.IDef _ ->
          let (x, ctx) = lower_path (xs, ts) ctx in
          (Mlir.EConst (Mlir.CFun x), ctx)
      | Mir.ITask _ -> todo ()
      | Mir.IVariant _ -> unreachable ()

and is_intrinsic d =
  d |> assoc_opt "intrinsic" |> Option.is_some

and resolve_intrinsic d (xs, ts) ctx =
  match d |> List.assoc_opt "intrinsic" with
  | Some Some Ast.LString y -> (y, ctx)
  | None -> lower_path (xs, ts) ctx
  | _ -> panic "Found non-string as intrinsic"

and lower_field_type (x, t) ctx =
  let (t, ctx) = lower_type t ctx in
  ((x, t), ctx)

and lower_field_expr (x, v) ctx =
  let fa = ctx |> Ctx.typeof_opt v |> Option.map (fun t -> (x, (v, t))) in
  (fa, ctx)

and lower_variant (xs, ts) ctx =
  match ctx.mir |> List.assoc (xs, ts) with
  | Mir.IVariant t ->
      let (t, ctx) = lower_type t ctx in
      let (xs, ctx) = lower_path (xs, ts) ctx in
      ((xs, t), ctx)
  | _ -> unreachable ()

and int_or_float t =
  match t with
  | t when Mir.is_int t -> Mlir.NInt
  | t when Mir.is_float t -> Mlir.NFlt
  | _ -> todo ()

and int_or_float_or_bool t =
  match t with
  | t when Mir.is_int t -> Mlir.EqInt
  | t when Mir.is_float t -> Mlir.EqFlt
  | t when Mir.is_bool t -> Mlir.EqBool
  | _ -> todo ()
