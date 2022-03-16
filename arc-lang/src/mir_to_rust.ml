open Mir
open Pretty
open Utils

let rec pr_mir (mir:Mir.mir) =
  let ctx = Ctx.brief_with_data mir in
  mir |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and pr_item_path (xs, ts) (ctx:Mir.mir Ctx.t) =
  let i = ctx.data |> assoc (xs, ts) in
  pr_item ((xs, ts), i) ctx;

and pr_variants xss ts ctx =
  pr_sep "" pr_variant (map (fun xs -> (xs, ts)) xss) (ctx |> Ctx.indent);
 
and pr_variant (xs, ts) (ctx:Mir.mir Ctx.t) =
  match ctx.data |> assoc (xs, ts) with
  | IVariant t ->
      pr_path (xs, ts) ctx;
      pr_paren (pr_type t) ctx
  | _ -> unreachable ()

and pr_item ((xs, ts), i) ctx =
  ctx |> pr_indent;
  match i with
  | IVal _ -> todo()
  | IEnum (_, xss) ->
      pr "#[arc_codegen::rewrite]";
      ctx |> pr_indent;
      pr "enum ";
      pr_path (xs, ts) ctx;
      pr " {";
      ctx |> Ctx.indent |> pr_indent;
      pr_variants xss ts ctx;
      ctx |> pr_indent;
      pr "}";
  | IExternDef _ -> ()
  | IExternType _ -> ()
  | IDef (_, ps, t, b) ->
      pr "fn ";
      pr_path (xs, ts) ctx;
      pr_params ps ctx;
      pr " -> ";
      pr_type t ctx;
      pr " ";
      pr_block b ctx;
  | ITask (_, _ps0, _ps1, _b) ->
      todo ()
  | IVariant _ -> ()

and pr_path (xs, ts) ctx =
  prr (xs |> String.concat "");
  ts |> List.iter (fun t -> pr_mangle_type t ctx)

and pr_interface (xs, ts) ctx =
  pr_path (xs, ts) ctx;
  pr_brack (pr_types ts) ctx;

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
      pr "fn";
      pr_paren (pr_list pr_type ts) ctx; 
      pr " -> ";
      pr_type t ctx;
  | TRecord _ -> pr_mangle_type t ctx
  | TNominal _ -> pr_mangle_type t ctx

and pr_mangle_type t ctx =
  let rec pr_mangle_type t =
    match t with
    | Mir.TNominal (xs, ts) ->
        pr_path (xs, ts) ctx;
    | Mir.TFunc (ts, t) ->
        pr "Func";
        pr_mangle_types ts;
        pr_mangle_type t;
        pr "End";
    | Mir.TRecord fts ->
        pr "Struct";
        pr_mangle_fields fts;
        pr "End";
  and pr_mangle_types ts =
    ts |> List.iter pr_mangle_type;
  and pr_mangle_field (x, t) =
    prr ((x |> String.length) |> Int.to_string);
    prr x;
    pr_mangle_type t;
  and pr_mangle_fields fts =
    fts |> List.iter pr_mangle_field
  in
  pr_mangle_type t

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
  | ECall (v, vs) ->
      pr_var v ctx;
      pr_paren (pr_list pr_var vs) ctx;
  | ECast (v, t) ->
      pr_var v ctx;
      pr " as ";
      pr_type t ctx;
  | EEmit (v0, v1) ->
      pr_var v0 ctx;
      pr ".push";
      pr_paren (pr_var v1) ctx;
      pr ".await?"
  | EReceive v ->
      pr_var v ctx;
      pr "pull().await?";
  | EEnwrap (xs, ts, v) ->
      pr "enwrap[";
      pr_path (xs, ts) ctx;
      if ts != [] then begin
        pr_brack (pr_list pr_type ts) ctx;
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
      pr "is!(";
      pr_path (xs, ts) ctx;
      if ts != [] then begin
        pr_brack (pr_list pr_type ts) ctx;
      end;
      pr ")";
      pr_paren (pr_var v) ctx;
  | ELit l ->
      pr_lit l ctx;
  | ELoop b ->
      pr "loop ";
      pr_block b ctx;
  | ERecord fvs ->
      pr "#{";
      pr_list pr_expr_field fvs ctx;
      pr "}";
  | EUnwrap (xs, ts, v) ->
      pr "unwrap[";
      pr_path (xs, ts) ctx;
      if ts != [] then begin
        pr_brack (pr_list pr_type ts) ctx;
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
      pr_path (xs, ts) ctx;

and pr_type_field (x, t) ctx =
  pr "pub ";
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_expr_field (x, v) ctx =
  pr_name x ctx;
  pr ": ";
  pr_var v ctx;

and pr_var x ctx =
  pr "val!";
  pr_paren (pr_name x) ctx
