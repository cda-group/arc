open Hir
open Utils

module Ctx = struct
  type t = {
    hir: Hir.hir;
    mir: Mir.mir;
    substs: subst list;
  }
  and subst = (Mir.name * Mir.ty) list

  let rec make (hir:Hir.hir) = { hir; mir = []; substs = [] }

  and has_instance s ctx =
    ctx.mir |> List.assoc_opt s |> Option.is_some

  (* Add a monomorphised instance *)
  and add_instance s i ctx =
    if not (ctx |> has_instance s) then
      { ctx with mir = (s, i)::ctx.mir }
    else
      ctx

  and push_subst s ctx =
    { ctx with substs = (s::ctx.substs) }

  and pop_subst ctx =
    { ctx with substs = tl ctx.substs }

  and substitute g ctx =
    ctx.substs |> hd |> assoc g

end

let rec mir_of_hir hir =
  let ctx = Ctx.make hir in
  let ctx = hir |> foldl lower_item ctx in
  ctx.mir

and lower_item ctx (xs, i) =
  match i with
  | Hir.IDef ([], ps, t, b) ->
      let (ps, ctx) = lower_params ps ctx in
      let (t, ctx) = lower_type t ctx in
      let (b, ctx) = lower_block b ctx in
      ctx |> Ctx.add_instance (xs, []) (Mir.IDef (ps, t, b))
  | Hir.ITask ([], ps, i0, i1, b) ->
      let (ps, ctx) = lower_params ps ctx in
      let (i0, ctx) = lower_interface i0 ctx in
      let (i1, ctx) = lower_interface i1 ctx in
      let (b, ctx) = lower_block b ctx in
      ctx |> Ctx.add_instance (xs, []) (Mir.ITask (ps, i0, i1, b))
  | Hir.IVal (t, b) ->
      let (t, ctx) = lower_type t ctx in
      let (b, ctx) = lower_block b ctx in
      ctx |> Ctx.add_instance (xs, []) (Mir.IVal (t, b))
  | _ -> ctx

and lower_interface (xs, ts) ctx =
  let (ts, ctx) = lower_types ts ctx in
  ((xs, ts), ctx)

and lower_block (ss, v) ctx =
  let (ss, ctx) = ss |> mapm lower_ssa ctx in
  ((ss, v), ctx)

and lower_ssa (v, t, e) ctx =
  let (t, ctx) = lower_type t ctx in
  let (e, ctx) = lower_expr e ctx in
  ((v, t, e), ctx)

and lower_expr (e:Hir.expr) (ctx:Ctx.t) =
  match e with
  | EAccess (v, x) ->
      (Mir.EAccess (v, x), ctx)
  | EEq (v0, v1) ->
      (Mir.EEq (v0, v1), ctx)
  | Hir.ECall (v, vs) ->
      (Mir.ECall (v, vs), ctx)
  | ECast (v, t) ->
      let (t, ctx) = lower_type t ctx in
      (Mir.ECast (v, t), ctx)
  | EEmit v ->
      (Mir.EEmit v, ctx)
  | EEnwrap (xs, ts, v) ->
      let (ts, ctx) = lower_types ts ctx in
      (Mir.EEnwrap (xs, ts, v), ctx)
  | Hir.EIf (v, b0, b1) ->
      let (b0, ctx) = lower_block b0 ctx in
      let (b1, ctx) = lower_block b1 ctx in
      (Mir.EIf (v, b0, b1), ctx)
  | EIs (xs, ts, v) ->
      let (ts, ctx) = lower_types ts ctx in
      (Mir.EIs (xs, ts, v), ctx)
  | ELit l ->
      (Mir.ELit l, ctx)
  | Hir.ELoop b ->
      let (b, ctx) = lower_block b ctx in
      (Mir.ELoop b, ctx)
  | Hir.EReceive -> (Mir.EReceive, ctx)
  | ERecord fvs -> (Mir.ERecord fvs, ctx)
  | EUnwrap (xs, ts, v) ->
      let (ts, ctx) = lower_types ts ctx in
      (Mir.EUnwrap (xs, ts, v), ctx)
  | EReturn v -> (Mir.EReturn v, ctx)
  | Hir.EBreak v -> (Mir.EBreak v, ctx)
  | Hir.EContinue -> (Mir.EContinue, ctx)
  | Hir.EItem (xs, ts) ->
      let (ts, ctx) = lower_types ts ctx in
      begin match ctx.hir |> assoc xs with
      | Hir.IDef (gs, ps, t, b) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let (ps, ctx) = lower_params ps ctx in
          let (t, ctx) = lower_type t ctx in
          let (b, ctx) = lower_block b ctx in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IDef (ps, t, b)) in
          let ctx = ctx |> Ctx.pop_subst in
          (Mir.EItem (xs, ts), ctx)
      | Hir.IExternDef (gs, ts1, t) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let (ts1, ctx) = lower_types ts1 ctx in
          let (t, ctx) = lower_type t ctx in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IExternDef (ts1, t)) in
          let ctx = ctx |> Ctx.pop_subst in
          (Mir.EItem (xs, ts), ctx)
      | Hir.ITask (gs, ps, i0, i1, b) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let (ps, ctx) = lower_params ps ctx in
          let (i0, ctx) = lower_interface i0 ctx in
          let (i1, ctx) = lower_interface i1 ctx in
          let (b, ctx) = lower_block b ctx in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.ITask (ps, i0, i1, b)) in
          let ctx = ctx |> Ctx.pop_subst in
          (Mir.EItem (xs, ts), ctx)
      | _ -> unreachable ()
      end

and lower_params ps (ctx:Ctx.t) =
  ps |> mapm lower_param ctx

and lower_param ((x, t):Hir.param) (ctx:Ctx.t) =
  let (t, ctx) = lower_type t ctx in
  ((x, t), ctx)

and lower_types ts (ctx:Ctx.t) =
  ts |> mapm lower_type ctx

and lower_row r ctx =
  let rec lower_row r ctx acc =
    match r with
    | Hir.TRowExtend ((x, t), r) ->
        let (t, ctx) = lower_type t ctx in
        lower_row r ctx ((x, t)::acc)
    | Hir.TRowEmpty -> (acc, ctx)
    | _ -> unreachable ()
  in
  let (fts, ctx) = lower_row r ctx [] in
  (fts |> List.rev, ctx)

and lower_type (t:Hir.ty) (ctx:Ctx.t) : (Mir.ty * Ctx.t) =
  match t with
  | Hir.TFunc (ts, t) ->
      let (ts, ctx) = lower_types ts ctx in
      let (t, ctx) = lower_type t ctx in
      (Mir.TFunc (ts, t), ctx)
  | Hir.TRecord t ->
      let (fts, ctx) = lower_row t ctx in
      (Mir.TRecord fts, ctx)
  | Hir.TRowEmpty -> unreachable ()
  | Hir.TRowExtend _ -> unreachable ()
  | Hir.TNominal (xs, ts) ->
      let (ts, ctx) = lower_types ts ctx in
      begin match ctx.hir |> assoc xs with
      | Hir.IExternType _ ->
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IExternType) in
          (Mir.TNominal (xs, ts), ctx)
      | Hir.IEnum (gs, xss) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let ctx = xss |> foldl (fun (ctx:Ctx.t) xs ->
            match ctx.hir |> assoc xs with
            | Hir.IVariant t ->
                let (t, ctx) = lower_type t ctx in
                ctx |> Ctx.add_instance (xs, ts) (Mir.IVariant t)
            | _ -> unreachable ()
          ) ctx in
          let ctx = ctx |> Ctx.pop_subst in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IEnum xss) in
          (Mir.TNominal (xs, ts), ctx)
      | _ -> unreachable ()
      end
  | Hir.TGeneric g ->
      let t = ctx |> Ctx.substitute g in
      (t, ctx)
  | Hir.TVar _ -> unreachable ()
