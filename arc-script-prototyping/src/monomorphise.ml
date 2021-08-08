open Hir
open Utils

module Ctx = struct
  type t = {
    hir: Hir.hir;
    instances: (signature * item) list
  }
  (* Here we should maybe mangle instead *)
  and signature = path * ty list

  let rec make (hir:Hir.hir) = { hir; instances = []; }

  and has_instance s ctx =
    ctx.instances |> List.assoc_opt s |> Option.is_some

  and add_instance s i ctx =
    if not (ctx |> has_instance s) then
      { ctx with instances = (s, i)::ctx.instances }
    else
      ctx
end

let rec monomorphise hir =
  let ctx = Ctx.make hir in
  let ctx = hir |> foldl mono_item ctx in
  ctx.instances |> map (fun ((xs, _), i) -> (xs, i))

and mono_item ctx (xs, i) =
  match i with
  | Hir.IFunc ([], ps, t, b) ->
      let ctx = mono_params ctx ps in
      let ctx = mono_type ctx t in
      let ctx = mono_block ctx b in
      ctx |> Ctx.add_instance (xs, []) i
  | Hir.IExternFunc ([], ps, t) ->
      let ctx = mono_params ctx ps in
      let ctx = mono_type ctx t in
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
      | Hir.IFunc (gs, ps, t, b) ->
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
          ctx |> Ctx.add_instance (xs, ts) (Hir.IFunc ([], ps, t, b))
      | Hir.IExternFunc (gs, ps, t) ->
          let s = zip gs ts in
          let f = Infer.instantiate s in
          let ps = tmap_params f ps in
          let t = f t in
          ctx |> Ctx.add_instance (xs, ts) (Hir.IExternFunc ([], ps, t))
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
          ctx |> Ctx.add_instance (xs, ts) (Hir.ITask ([], ps, (xs0, ts0), (xs1, ts1), b))
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
      | Hir.IExternType _ -> ctx |> Ctx.add_instance (xs, ts) (Hir.IExternType [])
      | Hir.IEnum (gs, xss) -> mono_enum ctx xs xss gs ts 
      | _ -> unreachable ()
      end
  | Hir.TGeneric _ -> unreachable ()
  | Hir.TArray t -> mono_type ctx t
  | Hir.TStream t -> mono_type ctx t
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
        ctx |> Ctx.add_instance (xs, ts) (Hir.IVariant t)
    | _ -> unreachable ()
  ) ctx in
  ctx |> Ctx.add_instance (xs, ts) (Hir.IEnum ([], xss))

