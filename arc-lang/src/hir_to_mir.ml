open Hir
open Utils

module Ctx = struct
  type t = {
    hir: Hir.hir;
    mir: Mir.mir;
  }

  let rec make (hir:Hir.hir) = { hir; mir = []; }

  and has_instance s ctx =
    ctx.mir |> List.assoc_opt s |> Option.is_some

  (* Add a monomorphised instance *)
  and add_instance s i ctx =
    if not (ctx |> has_instance s) then
      { ctx with mir = (s, i)::ctx.mir }
    else
      ctx
end

let rec mir_of_hir hir =
  let ctx = Ctx.make hir in
  let ctx = hir |> foldl mono_item ctx in
  ctx.mir

and mono_item ctx (xs, i) =
  match i with
  | Hir.IDef ([], ps, t, b) ->
      let ctx = mono_params ctx ps in
      let ctx = mono_type ctx t in
      let ctx = mono_block ctx b in
      ctx |> Ctx.add_instance (xs, []) i
  | Hir.ITask ([], ps, (xs0, ts0), (xs1, ts1), b) ->
      let ctx = mono_params ctx ps in
      let ctx = mono_types ctx ts0 in
      let ctx = mono_types ctx ts1 in
      let ctx = mono_block ctx b in
      let ctx = mono_enum_path ctx xs0 [] in
      let ctx = mono_enum_path ctx xs1 [] in
      ctx |> Ctx.add_instance (xs, []) i
  | _ -> ctx

and mono_block ctx (ss, _) =
  ss |> foldl mono_ssa ctx

and mono_ssa ctx (_, t, e) =
  Pretty_hir.debug_type t;
  print_endline "";
  print_endline "";
  let ctx = mono_type ctx t in
  mono_expr ctx e

and mono_expr ctx e =
  match e with
  | Hir.EAfter (_, b) -> mono_block ctx b
  | Hir.EEvery (_, b) -> mono_block ctx b
  | Hir.EIf (_, b0, b1) -> [b0; b1] |> foldl mono_block ctx
  | Hir.ELoop b -> mono_block ctx b
  | Hir.EItem (xs, ts) ->
      begin match ctx.hir |> assoc xs with
      | Hir.IDef (gs, ps, t, b) ->
          (* Instantiate function *)
          let s = zip gs ts in
          let f = Infer.instantiate s in
          let ps = tmap_params f ps in
          let t = f t in
          let b = tmap_block f b in
          (* Monomorphise function *)
          let ctx = mono_params ctx ps in
          let ctx = mono_type ctx t in
          let ctx = mono_block ctx b in
          ctx |> Ctx.add_instance (xs, ts) (Mir.IDef ([], ps, t, b))
      | Hir.IExternDef (gs, ts, t) ->
          let s = zip gs ts in
          let f = Infer.instantiate s in
          let ts = ts |> map f in
          let t = f t in
          ctx |> Ctx.add_instance (xs, ts) (Mir.IExternDef ([], ts, t))
      | Hir.ITask (gs, ps, (xs0, ts0), (xs1, ts1), b) ->
          (* Instantiate task *)
          let s = zip gs ts in
          let f = Infer.instantiate s in
          let ps = tmap_params f ps in
          let ts0 = ts0 |> map f in
          let ts1 = ts1 |> map f in
          let b = tmap_block f b in
          (* Monomorphise task *)
          let ctx = mono_params ctx ps in
          let ctx = mono_types ctx ts0 in
          let ctx = mono_types ctx ts1 in
          let ctx = mono_block ctx b in
          let ctx = mono_enum_path ctx xs0 [] in
          let ctx = mono_enum_path ctx xs1 [] in
          ctx |> Ctx.add_instance (xs, ts) (Mir.ITask ([], ps, (xs0, ts0), (xs1, ts1), b))
      | _ -> ctx
      end
  | _ -> ctx

and mono_params ctx ps =
  ps |> foldl mono_param ctx

and mono_param ctx p =
  mono_type ctx (p |> snd)

and mono_types ctx ts =
  ts |> foldl mono_type ctx

and mono_type ctx t = 
  match t with
  | Hir.TFunc (ts, t) -> mono_types ctx (t::ts)
  | Hir.TRecord t -> mono_type ctx t
  | Hir.TRowEmpty -> ctx
  | Hir.TRowExtend ((_, t), r) -> mono_types ctx [t; r]
  | Hir.TNominal (xs, ts) ->
      begin match ctx.hir |> assoc xs with
      | Hir.IExternType _ -> ctx |> Ctx.add_instance (xs, ts) (Mir.IExternType [])
      | Hir.IEnum (gs, xss) -> mono_enum ctx xs xss gs ts 
      | _ -> unreachable ()
      end
  | Hir.TGeneric _ -> unreachable ()
  | Hir.TVar _ -> unreachable ()

and mono_enum_path ctx xs ts =
  match ctx.hir |> assoc xs with
  | Hir.IEnum (gs, xss) -> mono_enum ctx xs xss gs ts
  | _ -> unreachable ()

and mono_enum ctx xs xss gs ts =
  (* Instantiate enum *)
  let s = zip gs ts in
  let ctx = xss |> foldl (fun (ctx:Ctx.t) xs ->
    match ctx.hir |> assoc xs with
    | Hir.IVariant t ->
        let t = Infer.instantiate s t in
        ctx |> Ctx.add_instance (xs, ts) (Mir.IVariant t)
    | _ -> unreachable ()
  ) ctx in
  ctx |> Ctx.add_instance (xs, ts) (Mir.IEnum ([], xss))
