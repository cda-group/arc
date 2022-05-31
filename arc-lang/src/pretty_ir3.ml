open Ir3
open Pretty
open Utils

let rec pr_ir3 (ir3:Ir3.ir3) =
  let ctx = Ctx.make in
  pr_items ir3 ctx;
  pr "\n";

and show_item (_, i) =
  match Ir3.item_loc i with
  | LocStd _ | NoLocStd when not !Args.show_std -> false
  | _ ->
    match i with
    | IExternType _ | IExternDef _ when not !Args.show_externs -> false
    | _ -> true

and pr_items is ctx =
  is |> filter show_item |> List.iter (fun i -> pr_item i ctx);

and pr_item ((xs, ts), i) ctx =
  ctx |> pr_indent;
  match i with
  | IVal (_, d, t, b) ->
      pr_decorator d ctx;
      pr "val ";
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b (ctx |> Ctx.indent);
  | IExternDef (_, d, a, ts, t) ->
      pr_decorator d ctx;
      pr "extern";
      if a then begin
        pr " async"
      end;
      pr " def ";
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr_paren (pr_types ts) ctx;
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType (_, d) ->
      pr_decorator d ctx;
      pr "extern type ";
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr ";";
  | IDef (_, d, ps, t, b) ->
      pr_decorator d ctx;
      pr "def ";
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr_params ps ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b ctx;
  | IType (_, d, t) ->
      pr_decorator d ctx;
      pr "type ";
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr " = ";
      pr_type t ctx;
      pr ";";

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

and pr_type_args ts ctx =
  if ts <> [] then begin
    pr_brack (pr_types ts) ctx;
  end

and pr_explicit_type_args ts ctx =
  if ts <> [] then begin
    pr "::";
    pr_brack (pr_list pr_type ts) ctx;
  end

and pr_types ts ctx =
  pr_list pr_type ts ctx;

and pr_type t ctx =
  match t with
  | TFunc (ts, t) ->
      pr "fun";
      pr_paren (pr_list pr_type ts) ctx; 
      pr ": ";
      pr_type t ctx;
  | TEnum xts ->
      pr_brace (pr_list (pr_variant pr_type) xts) ctx;
  | TRecord xts ->
      pr "#";
      pr_brace (pr_list (pr_field pr_type) xts) ctx;
  | TNominal (xs, ts) ->
      pr_path xs ctx;
      pr_type_args ts ctx;

and pr_explicit_block (ss, v) ctx =
  pr "{";
  let ctx' = ctx |> Ctx.indent in
  pr_sep "" pr_stmt ss ctx';
  ctx' |> pr_indent;
  pr_var v ctx;
  ctx |> pr_indent;
  pr "}";

and pr_block (ss, v) ctx =
  pr_explicit_block (ss, v) ctx
  (* begin match ss with *)
  (* | [] -> *)
  (*     pr_var v ctx; *)
  (* | _ -> *)
  (*     pr_explicit_block (ss, v) ctx *)
  (* end; *)

and pr_stmt s ctx =
  pr_indent ctx;
  begin match s with
  | Ir3.SVal (x, t, e) ->
      pr "val ";
      pr_name x ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_expr e ctx;
  end;
  pr ";"

and pr_name x _ctx =
  pr "%s" x;

and pr_var x ctx =
  pr_name x ctx;

and pr_expr e ctx =
  match e with
  | EUnwrap (_, x, v) ->
      pr "unwrap";
      pr_brack (pr_name x) ctx;
      pr_paren (pr_var v) ctx;
  | ECheck (_, x, v) ->
      pr "check";
      pr_brack (pr_name x) ctx;
      pr_paren (pr_var v) ctx;
  | EIf (_, v, b0, b1) ->
      pr "if ";
      pr_var v ctx;
      pr " ";
      pr_explicit_block b0 ctx;
      pr " else ";
      pr_explicit_block b1 ctx;
  | EAccess (_, e, x) ->
      pr_var e ctx;
      pr ".";
      pr_name x ctx;
  | EUpdate (_, v0, x, v1) ->
      pr_var v0 ctx;
      pr ".";
      pr_name x ctx;
      pr " = ";
      pr_var v1 ctx;
  | ECallExpr (_, e, vs) ->
      pr_var e ctx;
      pr_paren (pr_list pr_var vs) ctx;
  | ECallItem (_, xs, ts, vs) ->
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr_paren (pr_list pr_var vs) ctx;
  | ECast (_, e, t) ->
      pr_var e ctx;
      pr " as ";
      pr_type t ctx;
  | EEnwrap (_, x, e) ->
      pr "enwrap";
      pr_brack (pr_name x) ctx;
      pr_paren (pr_var e) ctx;
  | ELit (_, l) ->
      pr_lit l ctx;
  | ELoop (_, b) ->
      pr "loop ";
      pr_block b ctx;
  | ERecord (_, fvs) ->
      pr "#";
      pr_brace (pr_list (pr_field pr_var) fvs) ctx;
  | EReturn (_, e) ->
      pr "return ";
      pr_var e ctx;
  | EBreak (_, e) ->
      pr "break ";
      pr_var e ctx;
  | EContinue _ ->
      pr "continue"
  | EItem (_, xs, ts) ->
      pr_path xs ctx;
      pr_type_args ts ctx;
  | ESpawn (_, xs, ts, vs) ->
      pr "spawn ";
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr_paren (pr_list pr_var vs) ctx;

and pr_path xs ctx =
  if !Args.show_explicit_paths then
    begin
      pr "::";
      Pretty.pr_path xs ctx;
    end
  else
    pr "%s" (last xs);
