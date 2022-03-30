open Hir
open Pretty
open Utils

let rec pr_hir (hir:Hir.hir) debug =
  let ctx = if debug = Debug.Verbose then
    Ctx.verbose
  else
    Ctx.brief
  in
  hir |> filter (show_item ctx) |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and show_item (ctx:'a Ctx.t) (_, i) =
  match i with
  | IExternType _ | IExternDef _ when not ctx.show_externs -> false
  | _ -> true

and pr_item (xs, i) ctx =
  ctx |> pr_indent;
  match i with
  | IVal (d, t, b) ->
      pr_decorator d ctx;
      pr "val ";
      pr_path xs ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b (ctx |> Ctx.indent);
  | IEnum (d, gs, xss) ->
      pr_decorator d ctx;
      pr "enum ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr " {";
      (ctx |> Ctx.indent) |> pr_indent;
      pr_list pr_path xss (ctx |> Ctx.indent);
      ctx |> pr_indent;
      pr "}";
  | IExternDef (d, gs, ts, t) ->
      pr_decorator d ctx;
      pr "extern def ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_paren (pr_types ts) ctx;
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType (d, gs) ->
      pr_decorator d ctx;
      pr "extern type ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr ";";
  | IDef (d, gs, ps, t, b) ->
      pr_decorator d ctx;
      pr "def ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr ": ";
      pr_type t ctx;
      pr " ";
      pr_block b ctx;
  | ITask (d, gs, ps0, ps1, b) ->
      pr_decorator d ctx;
      pr "task ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_params ps0 ctx;
      pr ": ";
      pr_params ps1 ctx;
      pr " ";
      pr_block b ctx;
  | ITypeAlias (d, gs, t) ->
      pr_decorator d ctx;
      pr "type ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr " = ";
      pr_type t ctx;
      pr ";";
  | IVariant t ->
      pr "variant ";
      pr_path xs ctx;
      pr_paren (pr_type t) ctx;
      pr ";";
  | IClass _ -> ()
  | IClassDecl _ -> ()
  | IInstance _ -> ()
  | IInstanceDef _ -> ()

and pr_interface (xs, ts) ctx =
  pr_path xs ctx;
  pr_brack (pr_types ts) ctx;

and pr_generics gs ctx =
  if gs != [] then begin
    pr_brack (pr_list pr_generic gs) ctx;
  end

and pr_generic g ctx =
  pr_name g ctx;

and pr_params ps ctx =
  pr_paren (pr_list pr_param ps) ctx

and pr_param (x, t) ctx =
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_types ts ctx =
  pr_list pr_type ts ctx;

and pr_type t ctx =
  match t with
  | TFunc (ts, t) ->
      pr "fun";
      pr_paren (pr_list pr_type ts) ctx; 
      pr ": ";
      pr_type t ctx;
  | TRecord t ->
      pr "#";
      pr_brace (pr_type t) ctx;
  | TRowEmpty ->
      pr "∅"
  | TRowExtend ((x, t), r) ->
      pr_name x ctx;
      pr ": ";
      pr_type t ctx;
      begin match r with
      | TRowEmpty | TVar _ | TGeneric _ ->
          pr "|";
          pr_type r ctx;
      | _ ->
          pr ", ";
          pr_type r ctx;
      end
  | TNominal (xs, ts) ->
      pr_path xs ctx;
      if ts != [] then begin
        pr_brack (pr_types ts) ctx;
      end
  | TVar x -> pr "'%s" x;
  | TGeneric x -> pr "%s" x;

and pr_block (ss, v) ctx =
  let ctx' = ctx |> Ctx.indent in
  pr "{";
  if ss != [] then begin
    pr_sep ";" pr_ssa ss ctx';
    pr ";";
  end;
  ctx' |> pr_indent;
  pr_var v ctx';
  ctx |> pr_indent;
  pr "}";

and pr_ssa (x, t, e) ctx =
  ctx |> pr_indent;
  pr "val ";
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;
  pr " = ";
  pr_expr e ctx;

and pr_name x _ctx =
  pr "%s" x;

and pr_expr e ctx =
  match e with
  | EAccess (v, x) ->
      pr_var v ctx;
      pr ".";
      pr_name x ctx;
  | EUpdate (v0, x, v1) ->
      pr_var v0 ctx;
      pr ".";
      pr_name x ctx;
      pr " = ";
      pr_var v1 ctx;
  | ECall (v, vs) ->
      pr_var v ctx;
      pr_paren (pr_list pr_var vs) ctx;
  | ECast (v, t) ->
      pr_var v ctx;
      pr " as ";
      pr_type t ctx;
  | EReceive v ->
      pr "receive ";
      pr_var v ctx;
  | EEmit (v0, v1) ->
      pr_var v0 ctx;
      pr " ! ";
      pr_var v1 ctx;
  | EEnwrap (xs, ts, v) ->
      pr "enwrap[";
      pr_path xs ctx;
      if ts != [] then begin
        pr "[";
        pr_list pr_type ts ctx;
        pr "]";
      end;
      pr "]";
      pr_paren (pr_var v) ctx;
  | EIf (v, b0, b1) ->
      pr "if ";
      pr_var v ctx;
      pr " ";
      pr_block b0 ctx;
      pr " else ";
      pr_block b1 ctx;
  | EIs (xs, ts, v) ->
      pr "is[";
      pr_path xs ctx;
      if ts != [] then begin
        pr "[";
        pr_list pr_type ts ctx;
        pr "]";
      end;
      pr "]";
      pr_paren (pr_var v) ctx;
  | ELit l ->
      pr_lit l ctx;
  | ELoop b ->
      pr "loop ";
      pr_block b ctx;
  | EOn _ -> todo ()
  | ERecord fvs ->
      pr "#";
      pr_brace (pr_list (pr_field pr_var) fvs) ctx;
  | EUnwrap (xs, ts, v) ->
      pr "unwrap[";
      pr_path xs ctx;
      if ts != [] then begin
        pr "[";
        pr_list pr_type ts ctx;
        pr "]";
      end;
      pr "]";
      pr_paren (pr_var v) ctx;
  | EReturn v ->
      pr "return ";
      pr_var v ctx;
  | EBreak v ->
      pr "break ";
      pr_var v ctx;
  | EContinue ->
      pr "continue"
  | EItem (xs, ts) ->
      pr_path xs ctx;
      if ts != [] then begin
        pr "[";
        pr_list pr_type ts ctx;
        pr "]";
      end
