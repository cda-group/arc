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
  | Hir.IDef (a, [], ps, t, b) ->
      let (ps, ctx) = lower_params ps ctx in
      let (t, ctx) = lower_type t ctx in
      let (b, ctx) = lower_block b ctx in
      ctx |> Ctx.add_instance (xs, []) (Mir.IDef (a, ps, t, b))
  | Hir.ITask (a, [], ps0, ps1, b) ->
      let (ps0, ctx) = lower_params ps0 ctx in
      let (ps1, ctx) = lower_params ps1 ctx in
      let (b, ctx) = lower_block b ctx in
      ctx |> Ctx.add_instance (xs, []) (Mir.ITask (a, ps0, ps1, b))
  | Hir.IVal (a, t, b) ->
      let (t, ctx) = lower_type t ctx in
      let (b, ctx) = lower_block b ctx in
      ctx |> Ctx.add_instance (xs, []) (Mir.IVal (a, t, b))
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
  | Hir.EAccess (v, x) ->
      (Mir.EAccess (v, x), ctx)
  | Hir.EUpdate (v0, x, v1) ->
      (Mir.EUpdate (v0, x, v1), ctx)
  | Hir.ECall (v, vs) ->
      (Mir.ECall (v, vs), ctx)
  | Hir.ECast (v, t) ->
      let (t, ctx) = lower_type t ctx in
      (Mir.ECast (v, t), ctx)
  | Hir.EEmit (v0, v1) ->
      (Mir.EEmit (v0, v1), ctx)
  | Hir.EEnwrap (xs, ts, v) ->
      let (ts, ctx) = lower_types ts ctx in
      (Mir.EEnwrap (xs, ts, v), ctx)
  | Hir.EIf (v, b0, b1) ->
      let (b0, ctx) = lower_block b0 ctx in
      let (b1, ctx) = lower_block b1 ctx in
      (Mir.EIf (v, b0, b1), ctx)
  | Hir.EIs (xs, ts, v) ->
      let (ts, ctx) = lower_types ts ctx in
      (Mir.EIs (xs, ts, v), ctx)
  | Hir.ELit l ->
      (Mir.ELit l, ctx)
  | Hir.ELoop b ->
      let (b, ctx) = lower_block b ctx in
      (Mir.ELoop b, ctx)
  | Hir.EReceive v -> (Mir.EReceive v, ctx)
  | Hir.EOn _ -> todo ()
  | Hir.ERecord fvs -> (Mir.ERecord (fvs |> sort_expr_fields), ctx)
  | Hir.EUnwrap (xs, ts, v) ->
      let (ts, ctx) = lower_types ts ctx in
      (Mir.EUnwrap (xs, ts, v), ctx)
  | Hir.EReturn v -> (Mir.EReturn v, ctx)
  | Hir.EBreak v -> (Mir.EBreak v, ctx)
  | Hir.EContinue -> (Mir.EContinue, ctx)
  | Hir.EItem (xs, ts) ->
      let (ts, ctx) = lower_types ts ctx in
      begin match ctx.hir |> assoc xs with
      | Hir.IDef (a, gs, ps, t, b) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let (ps, ctx) = lower_params ps ctx in
          let (t, ctx) = lower_type t ctx in
          let (b, ctx) = lower_block b ctx in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IDef (a, ps, t, b)) in
          let ctx = ctx |> Ctx.pop_subst in
          (Mir.EItem (xs, ts), ctx)
      | Hir.IExternDef (a, gs, ts1, t) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let (ts1, ctx) = lower_types ts1 ctx in
          let (t, ctx) = lower_type t ctx in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IExternDef (a, ts1, t)) in
          let ctx = ctx |> Ctx.pop_subst in
          (Mir.EItem (xs, ts), ctx)
      | Hir.ITask (a, gs, ps0, ps1, b) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let (ps0, ctx) = lower_params ps0 ctx in
          let (ps1, ctx) = lower_params ps1 ctx in
          let (b, ctx) = lower_block b ctx in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.ITask (a, ps0, ps1, b)) in
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
    | Hir.TGeneric g ->
      begin
        match ctx |> Ctx.substitute g with
        | Mir.TRecord fts -> (acc @ fts, ctx)
        | _ -> unreachable ()
      end
    | _ -> unreachable ()
  in
  let (fts, ctx) = lower_row r ctx [] in
  (fts |> List.rev |> sort_type_fields, ctx)

and sort_expr_fields fts =
  fts |> List.sort (fun (a, _) (b, _) -> String.compare a b)

and sort_type_fields fts =
  fts |> List.sort (fun (a, _) (b, _) -> String.compare a b)

and lower_type (t:Hir.ty) (ctx:Ctx.t) : (Mir.ty * Ctx.t) =
  match t with
  | Hir.TFunc (ts, t) ->
      let (ts, ctx) = lower_types ts ctx in
      let (t, ctx) = lower_type t ctx in
      (Mir.TFunc (ts, t), ctx)
  | Hir.TRecord t ->
      let (fts, ctx) = lower_row t ctx in
      (Mir.TRecord fts, ctx)
  | Hir.TRowEmpty | Hir.TRowExtend _ ->
      let (fts, ctx) = lower_row t ctx in
      (Mir.TRecord fts, ctx)
  | Hir.TNominal (xs, ts) ->
      let (ts, ctx) = lower_types ts ctx in
      begin match ctx.hir |> assoc xs with
      | Hir.IExternType (a, _) ->
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IExternType a) in
          (Mir.TNominal (xs, ts), ctx)
      | Hir.IEnum (a, gs, xss) ->
          let ctx = ctx |> Ctx.push_subst (zip gs ts) in
          let ctx = xss |> foldl (fun (ctx:Ctx.t) xs ->
            match ctx.hir |> assoc xs with
            | Hir.IVariant t ->
                let (t, ctx) = lower_type t ctx in
                ctx |> Ctx.add_instance (xs, ts) (Mir.IVariant t)
            | _ -> unreachable ()
          ) ctx in
          let ctx = ctx |> Ctx.pop_subst in
          let ctx = ctx |> Ctx.add_instance (xs, ts) (Mir.IEnum (a, xss)) in
          (Mir.TNominal (xs, ts), ctx)
      | _ -> unreachable ()
      end
  | Hir.TGeneric g ->
      let t = ctx |> Ctx.substitute g in
      (t, ctx)
  | Hir.TVar _ -> panic "Tried to lower a type variable in HIR => MIR"
