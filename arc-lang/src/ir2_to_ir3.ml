open Utils
open Ir2

module NameSet = Set.Make(struct type t = name let compare = compare end)
module PathMap = Map.Make(struct type t = path let compare = compare end)

module Ctx = struct
  type t = {
    ir2: ir2;
    ir3: Ir3.ir3;
    istack: iscope list;
  }
  and iscope = {
    isubst: (Ir2.generic * Ir3.ty) list;
  }

  let make ir2 = {
    ir2;
    ir3 = [];
    istack = [];
  }

  and push_iscope gs ts ctx =
    let isubst = zip gs ts in
    { ctx with istack = { isubst }::ctx.istack }

  and pop_iscope ctx =
    { ctx with istack = tl ctx.istack }

  and add_item (xs, ts) i ctx =
    { ctx with ir3 = (((xs, ts), i)::ctx.ir3) }

  and has_item (xs, ts) ctx =
    ctx.ir3 |> assoc_opt (xs, ts) |> Option.is_some

  and find_g g ctx =
    let iscope = hd ctx.istack in
    iscope.isubst |> assoc_opt g |> Option.get

end

let rec ir2_to_ir3 ir1 =
  let ctx = Ctx.make ir1 in
  let ctx = ir1 |> foldl (fun ctx i -> ctx |> lower_item i) ctx in
  ctx.ir3 |> rev

and lower_block (ss, v) ctx =
  let (ss, ctx) = ss |> mapm lower_stmt ctx in
  let b = (ss, v) in
  (b, ctx)

and lower_stmt s ctx =
  match s with
  | SVal (x, t, e) ->
      let (t, ctx) = ctx |> lower_type t in
      let (e, ctx) = ctx |> lower_expr e in
      let s = Ir3.SVal (x, t, e) in
      (s, ctx)

and lower_param (x, t) ctx =
  let (t, ctx) = ctx |> lower_type t in
  ((x, t), ctx)

and lower_item (xs, i) ctx =
  match i with
  | IDef (loc, d, [], ps, t, b) ->
      let (ps, ctx) = ps |> mapm lower_param ctx in
      let (t, ctx) = ctx |> lower_type t in
      let (b, ctx) = ctx |> lower_block b in
      ctx |> Ctx.add_item (xs, []) (Ir3.IDef (loc, d, ps, t, b))
  | IExternDef (loc, d, async, [], ts, t) ->
      let (t, ctx) = ctx |> lower_type t in
      let (ts, ctx) = ts |> mapm lower_type ctx in
      ctx |> Ctx.add_item (xs, []) (Ir3.IExternDef (loc, d, async, ts, t))
  | IExternType (loc, d, []) ->
      ctx |> Ctx.add_item (xs, []) (Ir3.IExternType (loc, d))
  | IType (loc, d, [], t) ->
      let (t, ctx) = ctx |> lower_type t in
      ctx |> Ctx.add_item (xs, []) (Ir3.IType (loc, d, t))
  | IVal (loc, d, t, b) ->
      let (t, ctx) = ctx |> lower_type t in
      let (b, ctx) = ctx |> lower_block b in
      ctx |> Ctx.add_item (xs, []) (Ir3.IVal (loc, d, t, b))
  | IClass _ | IClassDef _ | IInstance _ | IInstanceDef _ -> todo () 
  | _ -> ctx

and lower_type t ctx =
  match t with
  | TFunc (ts, t) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let (t, ctx) = ctx |> lower_type t in
      (Ir3.TFunc (ts, t), ctx)
  | TRecord r ->
      let (r, ctx) = ctx |> lower_row r in
      (Ir3.TRecord r, ctx)
  | TEnum r ->
      let (r, ctx) = ctx |> lower_row r in
      (Ir3.TEnum r, ctx)
  | TRowEmpty | TRowExtend _ ->
      let (xts, ctx) = ctx |> lower_row t in
      (Ir3.TRecord xts, ctx)
  | TNominal (xs, ts) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let t = Ir3.TNominal (xs, ts) in
      let ctx = ctx |> lower_type_item xs ts in
      (t, ctx)
  | TGeneric g ->
      let t = ctx |> Ctx.find_g g in
      (t, ctx)

and lower_row r ctx =
  let rec lower_row r acc ctx =
    match r with
    | TRowExtend ((x, t), r) ->
        let (t, ctx) = ctx |> lower_type t in
        ctx |> lower_row r ((x, t)::acc)
    | TRowEmpty -> (acc, ctx)
    | TGeneric g ->
      begin
        match ctx |> Ctx.find_g g with
        | Ir3.TRecord xts -> (acc @ xts, ctx)
        | Ir3.TEnum xts -> (acc @ xts, ctx)
        | _ -> unreachable ()
      end
    | _ -> unreachable ()
  in
  let (xts, ctx) = ctx |> lower_row r [] in
  (xts |> List.rev |> sort_type_fields, ctx)

and sort_expr_fields xvs =
  xvs |> List.sort (fun (a, _) (b, _) -> String.compare a b)

and sort_type_fields xvs =
  xvs |> List.sort (fun (a, _) (b, _) -> String.compare a b)

and lower_expr e ctx =
  match e with
  | EAccess (loc, v1, x) ->
      let e = Ir3.EAccess (loc, v1, x) in
      (e, ctx)
  | ESubset _ -> todo ()
  | EUpdate (loc, v1, x, v2) ->
      let e = Ir3.EUpdate (loc, v1, x, v2) in
      (e, ctx)
  | ECast (loc, v1, t2) ->
      let (t2, ctx) = ctx |> lower_type t2 in
      let e = Ir3.ECast (loc, v1, t2) in
      (e, ctx)
  | ELit (loc, l) ->
      let e = Ir3.ELit (loc, l) in
      (e, ctx)
  | ELoop (loc, b) ->
      let (b, ctx) = ctx |> lower_block b in
      let e = Ir3.ELoop (loc, b) in
      (e, ctx)
  | ERecord (loc, (xvs, _)) ->
      let xvs = sort_expr_fields xvs in
      let e = Ir3.ERecord (loc, xvs) in
      (e, ctx)
  | EIf (loc, v, b0, b1) ->
      let (b0, ctx) = ctx |> lower_block b0 in
      let (b1, ctx) = ctx |> lower_block b1 in
      let e = Ir3.EIf (loc, v, b0, b1) in
      (e, ctx)
  | EReturn (loc, v) ->
      let e = Ir3.EReturn (loc, v) in
      (e, ctx)
  | EBreak (loc, v) ->
      let e = Ir3.EBreak (loc, v) in
      (e, ctx)
  | EContinue (loc) ->
      let e = Ir3.EContinue (loc) in
      (e, ctx)
  | EEnwrap (loc, x, v1) ->
      let e = Ir3.EEnwrap (loc, x, v1) in
      (e, ctx)
  | EUnwrap (loc, x, v1) ->
      let e = Ir3.EUnwrap (loc, x, v1) in
      (e, ctx)
  | ECheck (loc, x, v1) ->
      let e = Ir3.ECheck (loc, x, v1) in
      (e, ctx)
  | EItem (loc, xs, ts) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let ctx = ctx |> lower_expr_item xs ts in
      let e = Ir3.EItem (loc, xs, ts) in
      (e, ctx)
  | ECallExpr (loc, v, vs) ->
      let e = Ir3.ECallExpr (loc, v, vs) in
      (e, ctx)
  | ECallItem (loc, xs, ts, vs) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let ctx = ctx |> lower_expr_item xs ts in
      let e = Ir3.ECallItem (loc, xs, ts, vs) in
      (e, ctx)
  | ESpawn (loc, xs, ts, vs) ->
      let (ts, ctx) = ts |> mapm lower_type ctx in
      let ctx = ctx |> lower_expr_item xs ts in
      let e = Ir3.ESpawn (loc, xs, ts, vs) in
      (e, ctx)

and lower_expr_item xs ts ctx =
  if not (ctx |> Ctx.has_item (xs, ts)) then
    match ctx.ir2 |> assoc xs with
    | IDef (loc, d, gs, ps, t, b) ->
        let ctx = ctx |> Ctx.push_iscope gs ts in
        let (ps, ctx) = ps |> mapm lower_param ctx in
        let (t, ctx) = ctx |> lower_type t in
        let (b, ctx) = ctx |> lower_block b in
        let ctx = ctx |> Ctx.add_item (xs, ts) (Ir3.IDef (loc, d, ps, t, b)) in
        let ctx = ctx |> Ctx.pop_iscope in
        ctx
    | IExternDef (loc, d, a, gs, ts1, t) ->
        let ctx = ctx |> Ctx.push_iscope gs ts in
        let (ts1, ctx) = ts1 |> mapm lower_type ctx in
        let (t, ctx) = ctx |> lower_type t in
        let ctx = ctx |> Ctx.add_item (xs, ts) (Ir3.IExternDef (loc, d, a, ts1, t)) in
        let ctx = ctx |> Ctx.pop_iscope in
        ctx
    | _ ->
        unreachable ()
  else
    ctx

and lower_type_item xs ts (ctx:Ctx.t) =
  if not (ctx |> Ctx.has_item (xs, ts)) then
    match ctx.ir2 |> assoc xs with
    | IExternType (loc, d, _) ->
        ctx |> Ctx.add_item (xs, ts) (Ir3.IExternType (loc, d))
    | _ -> unreachable ()
  else
    ctx

