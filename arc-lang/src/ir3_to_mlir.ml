open Utils
open Ir3

module Ctx = struct
  type t = {
    ir3: ir3;
    mlir: Mlir.mlir;
    vstack: vscope list;
    next_ssa_uid: Gen.t;
  }
  and vscope = {
    ssas: Mlir.ssa list;
    vsubsts: (Ir3.name * Mlir.arg) list;
  }

  let rec make ir3 = {
    ir3;
    mlir = [];
    vstack = [];
    next_ssa_uid = Gen.make ();
  }

  and fresh_v ctx =
    let (n, next_ssa_uid) = ctx.next_ssa_uid |> Gen.fresh in
    let x = Printf.sprintf "v%d" n in
    let ctx = { ctx with next_ssa_uid; } in
    (x, ctx)

  and push_vscope ctx =
    { ctx with vstack = { ssas = []; vsubsts = []; }::ctx.vstack }

  and pop_vscope ctx =
    let vscope = ctx.vstack |> hd in
    (vscope.ssas |> rev, { ctx with vstack = tl ctx.vstack })

  and bind_param (v, t) ctx =
    let (v', ctx) = ctx |> fresh_v in
    let vscope = ctx.vstack |> hd in
    let vstack = ctx.vstack |> tl in
    let vstack = { vscope with vsubsts = (v, (v', t))::vscope.vsubsts }::vstack in
    let ctx = { ctx with vstack } in
    ((v', t), ctx)

  and add_ssa p o ctx =
    let vscope = ctx.vstack |> hd in
    let vstack = ctx.vstack |> tl in
    { ctx with vstack = { vscope with ssas = (p, o)::vscope.ssas }::vstack }

  and add_item x i ctx =
    { ctx with mlir = (x, i)::ctx.mlir }

  and find_v_opt v ctx =
    ctx.vstack |> List.find_map (fun vscope -> vscope.vsubsts |> List.assoc_opt v)

  and find_v v ctx =
    ctx |> find_v_opt v |> Option.get

end

let rec ir3_to_mlir ir1 =
  let (ctx:Ctx.t) = Ctx.make ir1 in
  let ctx = ir1 |> foldl (fun ctx i -> ctx |> lower_item i) ctx in
  ctx.mlir |> rev

and lower_block (ss, v) terminator ctx : (Mlir.block * Ctx.t) =
  let ctx = ctx |> Ctx.push_vscope in
  let (ctx:Ctx.t) = ss |> foldl (fun ctx s -> lower_stmt s ctx) ctx in
  let ctx = terminator v ctx in
  let (b, ctx) = ctx |> Ctx.pop_vscope in
  (b, ctx)

and lower_var v ctx =
  let a = ctx |> Ctx.find_v v in
  (a, ctx)

and lower_var_opt v ctx =
  let a = ctx |> Ctx.find_v_opt v in
  (a, ctx)

and lower_stmt (SVal (v, t, e)) ctx : Ctx.t =
  let (p, ctx) = lower_param_opt (v, t) ctx in
  match e with
  | EAccess (loc, v1, x1) ->
      let (a, ctx) = ctx |> lower_var v1 in
      ctx |> Ctx.add_ssa p (Mlir.OAccess (loc, a, x1))
  | EBreak (loc, v) ->
      let (a, ctx) = ctx |> lower_var_opt v in
      ctx |> Ctx.add_ssa p (Mlir.OBreak (loc, a))
  | ECallExpr (loc, v, vs) ->
      let (a, ctx) = ctx |> lower_var v in
      let (args, ctx) = vs |> mapm_filter lower_var_opt ctx in
      ctx |> Ctx.add_ssa p (Mlir.OCallExpr (loc, a, args))
  | ECallItem (loc, xs, ts, vs) ->
      let (x, ctx) = lower_path (xs, ts) ctx in
      let (args, ctx) = vs |> mapm_filter lower_var_opt ctx in
      ctx |> Ctx.add_ssa p (Mlir.OCallItem (loc, x, args))
  | ECast _ ->
      todo ()
  | EContinue loc ->
      ctx |> Ctx.add_ssa p (Mlir.OContinue loc)
  | EEnwrap (loc, x, v) ->
      let (a, ctx) = ctx |> lower_var_opt v in
      ctx |> Ctx.add_ssa p (Mlir.OEnwrap (loc, x, a))
  | EUnwrap (loc, x, v) ->
      let (a, ctx) = ctx |> lower_var v in
      ctx |> Ctx.add_ssa p (Mlir.OUnwrap (loc, x, a))
  | ECheck (loc, x, v) ->
      let (a, ctx) = ctx |> lower_var v in
      ctx |> Ctx.add_ssa p (Mlir.OCheck (loc, x, a))
  | EItem (loc, xs, ts) ->
      let (x, ctx) = lower_path (xs, ts) ctx in
      ctx |> Ctx.add_ssa p (Mlir.OConst (loc, Mlir.CFun x))
  | EIf (loc, v, b0, b1) ->
      let (a, ctx) = ctx |> lower_var v in
      let (b0, ctx) = ctx |> lower_block b0 result in
      let (b1, ctx) = ctx |> lower_block b1 result in
      ctx |> Ctx.add_ssa p (Mlir.OIf (loc, a, b0, b1))
  | ELit (loc, l) ->
      begin match l with
      | Ast.LInt (_, d, _) ->
          ctx |> Ctx.add_ssa p (Mlir.OConst (loc, Mlir.CInt d))
      | Ast.LFloat (_, f, _) ->
          ctx |> Ctx.add_ssa p (Mlir.OConst (loc, Mlir.CFloat f))
      | Ast.LBool (_, b) ->
          ctx |> Ctx.add_ssa p (Mlir.OConst (loc, Mlir.CBool b))
      | Ast.LString (_, s) ->
          ctx |> Ctx.add_ssa p (Mlir.OConst (loc, Mlir.CAdt (Printf.sprintf "\\\"%s\\\"" s)))
      | Ast.LUnit _ ->
          ctx
      | Ast.LChar (_, c) ->
          ctx |> Ctx.add_ssa p (Mlir.OConst (loc, Mlir.CAdt (Printf.sprintf "'%c'" c)))
      end
  | ELoop (loc, b) ->
      let (b, ctx) = ctx |> lower_block b yield in
      ctx |> Ctx.add_ssa p (Mlir.OLoop (loc, b))
  | ERecord (loc, vts) ->
      let (vts, ctx) = vts |> mapm_filter lower_field_expr_opt ctx in
      let (xs, ts) = vts |> map snd |> Utils.unzip in
      ctx |> Ctx.add_ssa p (Mlir.ORecord (loc, xs, ts))
  | EReturn (loc, v) ->
      let (a, ctx) = ctx |> lower_var_opt v in
      ctx |> Ctx.add_ssa p (Mlir.OReturn (loc, a))
  | EUpdate (loc, v0, x, v1) ->
      let (a0, ctx) = ctx |> lower_var v0 in
      let (a1, ctx) = ctx |> lower_var v1 in
      ctx |> Ctx.add_ssa p (Mlir.OUpdate (loc, a0, x, a1))
  | ESpawn (loc, xs, ts, vs) ->
      let (x, ctx) = lower_path (xs, ts) ctx in
      let (args, ctx) = vs |> mapm_filter lower_var_opt ctx in
      ctx |> Ctx.add_ssa p (Mlir.OSpawn (loc, x, args))

and lower_param_opt (v, t) (ctx:Ctx.t) =
  if Ir3.is_unit t then
    (None, ctx)
  else
    let (t, ctx) = lower_type t ctx in
    let (v, ctx) = ctx |> Ctx.bind_param (v, t) in
    (Some v, ctx)

and lower_field_type_opt (x, t) ctx =
  lower_param_opt (x, t) ctx

and lower_field_expr_opt (x, v) ctx =
  let (v, ctx) = lower_var_opt v ctx in
  match v with
  | None -> (None, ctx)
  | Some v -> (Some (x, v), ctx)

and return v ctx =
  let a = ctx |> Ctx.find_v_opt v in
  ctx |> Ctx.add_ssa None (Mlir.OReturn (NoLoc, a))

and return_none _ ctx =
  ctx |> Ctx.add_ssa None (Mlir.OReturn (NoLoc, None))

and result v ctx =
  let a = ctx |> Ctx.find_v_opt v in
  ctx |> Ctx.add_ssa None (Mlir.OResult (NoLoc, a))

and yield _ ctx =
  ctx |> Ctx.add_ssa None (Mlir.OYield NoLoc)

and lower_item ((xs, ts), i) ctx =
  match i with
  | IDef (loc, _, ps, t, b) ->
      let (x, ctx) = lower_path (xs, ts) ctx in
      let ctx = ctx |> Ctx.push_vscope in
      let (ps, ctx) = ps |> mapm_filter lower_param_opt ctx in
      let (t, ctx) = ctx |> lower_type_opt t in
      let (b, ctx) = ctx |> lower_block b return in
      let (_, ctx) = ctx |> Ctx.pop_vscope in
      ctx |> Ctx.add_item x (Mlir.IFunc (loc, ps, t, b))
  | IExternDef (loc, d, async, ts, t) ->
      if is_defined_in_mlir d then
        ctx
      else
        let (x, ctx) = lower_path (xs, ts) ctx in
        let (x_rust, ctx) = lower_item_extern_def_path x d ctx in
        let (ts, ctx) = ts |> filter_unit |> mapm lower_type ctx in
        let ps = ts |> Ir1.indexes_to_fields in
        let (t, ctx) = lower_type t ctx in
        ctx |> Ctx.add_item x (Mlir.IExternFunc (loc, x_rust, async, ps, t))
  | IVal _ ->
      todo ()
  | IExternType _ | IType _ -> ctx

and is_defined_in_mlir d =
  d |> assoc_opt "mlir" |> Option.is_some

and lower_expr_extern_def_path d (xs, ts) ctx =
  match d |> List.assoc_opt "mlir" with
  | Some Some Ast.LString (_, x) -> (x, ctx)
  | None -> lower_path (xs, ts) ctx
  | _ -> panic "Found non-string as mlir"

and lower_item_extern_def_path x d ctx =
  match d |> List.assoc_opt "rust" with
  | Some Some Ast.LString (_, x) -> (x, ctx)
  | None -> panic (Printf.sprintf "rust attribute must be specified for %s" x)
  | _ -> panic (Printf.sprintf "Found non-string as rust attribute-value for %s" x)

and filter_unit ts =
  ts |> filter (fun t -> not (t |> Ir3.is_unit))

and lower_type_field (x, t) ctx =
  let (t, ctx) = ctx |> lower_type t in
  ((x, t), ctx)

and lower_type_opt t ctx =
  if Ir3.is_unit t then
    (None, ctx)
  else
    let (t, ctx) = lower_type t ctx in
    (Some t, ctx)

and lower_type t ctx =
  match t with
  | TFunc (ts, t) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = ctx |> lower_type t in
      (Mlir.TFunc (ts, t), ctx)
  | TEnum xts ->
      let (xts, ctx) = xts |> mapm lower_type_field ctx in
      (Mlir.TEnum xts, ctx)
  | TRecord xts ->
      let (xts, ctx) = xts |> mapm lower_type_field ctx in
      (Mlir.TRecord xts, ctx)
  | TNominal (xs, ts) ->
      ctx |> lower_type_item xs ts

and lower_type_item xs ts ctx =
  begin match ctx.ir3 |> List.assoc (xs, ts) with
  | Ir3.IExternType (_, d) ->
    begin match d |> List.assoc_opt "mlir" with
    | Some Some Ast.LString (_, x) ->
        (Mlir.TNative x, ctx)
    | None ->
        begin match d |> List.assoc_opt "rust" with
        | Some Some Ast.LString (_, x) ->
            let (ts, ctx) = ts |> mapm lower_type ctx in
            (Mlir.TAdt (x, ts), ctx)
        | _ -> panic "Expected literal string, got something else"
        end
    | _ -> panic "Expected mlir or rust attribute"
    end
  | _ -> unreachable ()
  end

and lower_path (xs, ts) ctx =
  (* Mangle path *)
  let (ts, ctx) = ts |> mapm lower_type ctx in
  let s0 = xs |> String.concat "_" in
  let s1 = ts |> map mangle_type |> String.concat "" in
  (s0 ^ s1, ctx)

and mangle_type t =
  let rec mangle_type t acc =
    match t with
    | Mlir.TAdt (x, ts) ->
        let acc = "Adt"::acc in
        let acc = mangle_types ts acc in
        let acc = x::acc in
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
