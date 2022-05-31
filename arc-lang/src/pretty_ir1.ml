open Ir1
open Pretty
open Utils

let rec pr_ir1 (ir1:Ir1.ir1) =
  let ctx = Ctx.make in
  pr_items ir1 ctx;
  pr "\n";

and show_item (_, i) =
  match Ir1.item_loc i with
  | LocStd _ | NoLocStd when not !Args.show_std -> false
  | _ ->
    match i with
    | IExternType _ | IExternDef _ when not !Args.show_externs -> false
    | _ -> true

and pr_items is ctx =
  is |> filter show_item |> List.iter (fun i -> pr_item i ctx);

and pr_item (xs, i) ctx =
  ctx |> pr_indent;
  match i with
  | IVal (_, d, t, b) ->
      pr_decorator d ctx;
      pr "val ";
      pr_path xs ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b (ctx |> Ctx.indent);
  | IExternDef (_, d, a, gs, ts, t, _bs) ->
      pr_decorator d ctx;
      pr "extern";
      if a then begin
        pr " async"
      end;
      pr " def ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_paren (pr_types ts) ctx;
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType (_, d, gs, _bs) ->
      pr_decorator d ctx;
      pr "extern type ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr ";";
  | IDef (_, d, gs, ps, t, _bs, b) ->
      pr_decorator d ctx;
      pr "def ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b ctx;
  | IType (_, d, gs, t, _bs) ->
      pr_decorator d ctx;
      pr "type ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr " = ";
      pr_type t ctx;
      pr ";";
  | IClass _ -> ()

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
  | TRecord t ->
      pr "#";
      pr_brace (pr_row_field t) ctx
  | TEnum t ->
      pr "enum ";
      pr_brace (pr_row_variant t) ctx
  | TRowExtend ((x, t), r) ->
      pr "TRowExtend((";
      pr_name x ctx;
      pr ", ";
      pr_type t ctx;
      pr "), ";
      pr_type r ctx;
      pr ")";
  | TRowEmpty -> pr "∅"
  | TNominal (xs, ts) ->
      pr_path xs ctx;
      pr_type_args ts ctx;
  | TVar x -> pr "'%s" x;
  | TGeneric x -> pr "%s" x;
  | TInverse t ->
      pr "Inverse";
      pr_paren (pr_type t) ctx;

and pr_row_field t ctx =
  match t with
  | TRowEmpty ->
      pr "∅"
  | TRowExtend ((x, t), r) ->
      pr_name x ctx;
      pr ": ";
      pr_type t ctx;
      begin match r with
      | TRowEmpty | TGeneric _ -> pr "|";
      | _ -> pr ", ";
      end;
      pr_row_field r ctx;
  | _ -> pr_type t ctx

and pr_row_variant t ctx =
  match t with
  | TRowEmpty ->
      pr "∅"
  | TRowExtend ((x, t), r) ->
      pr_name x ctx;
      pr_type t ctx;
      begin match r with
      | TRowEmpty | TGeneric _ -> pr "|";
      | _ -> pr ", ";
      end;
      pr_row_variant r ctx;
  | _ -> pr_type t ctx


and pr_explicit_block (ss, e) ctx =
  pr "{";
  let ctx' = ctx |> Ctx.indent in
  pr_sep "" pr_stmt ss ctx';
  ctx' |> pr_indent;
  pr_expr e ctx;
  ctx |> pr_indent;
  pr "}";

and pr_block (ss, e) ctx =
  (* begin match ss with *)
  (* | [] -> *)
  (*     pr_expr e ctx; *)
  (* | _ -> *)
      pr_explicit_block (ss, e) ctx
  (* end; *)

and pr_stmt s ctx =
  pr_indent ctx;
  begin match s with
  | Ir1.SExpr e ->
      pr_expr e ctx;
  end;
  pr ";"

and pr_name x _ctx =
  pr "%s" x;

and pr_expr e ctx =
  let pr_expr e ctx =
    match e with
    | EAccess (_, _, e, x) ->
        pr_expr e ctx;
        pr ".";
        pr_name x ctx;
    | EUpdate (_, _, v0, x, v1) ->
        pr_expr v0 ctx;
        pr ".";
        pr_name x ctx;
        pr " = ";
        pr_expr v1 ctx;
    | ECallExpr (_, _, e, vs) ->
        pr_expr e ctx;
        pr_paren (pr_list pr_expr vs) ctx;
    | ECallItem (_, _, xs, ts, vs) ->
        pr_path xs ctx;
        pr_type_args ts ctx;
        pr_paren (pr_list pr_expr vs) ctx;
    | ECast (_, _, e, t) ->
        pr_expr e ctx;
        pr " as ";
        pr_type t ctx;
    | EEnwrap (_, _, x, e) ->
        pr "enwrap";
        pr_brack (pr_name x) ctx;
        pr_paren (pr_expr e) ctx;
    | ELit (_, _, l) ->
        pr_lit l ctx;
    | ELoop (_, _, b) ->
        pr "loop ";
        pr_block b ctx;
    | EOn _ -> todo ()
    | ERecord (_, _, fvs) ->
        pr_record_explicit pr_expr fvs ctx
    | EReturn (_, _, e) ->
        pr "return ";
        pr_expr e ctx;
    | EBreak (_, _, e) ->
        pr "break ";
        pr_expr e ctx;
    | EContinue _ ->
        pr "continue"
    | EItem (_, _, xs, ts) ->
        pr_path xs ctx;
        pr_explicit_type_args ts ctx;
    | EVar (_, _, x) ->
        pr_name x ctx;
    | EMatch (_, _, e, arms) ->
        begin match arms with
        | [(p, (ss, e1))] ->
          pr "val ";
          pr_pat p ctx;
          pr " = ";
          pr_expr e ctx;
          pr ";";
          (* let ctx = ctx |> Ctx.indent in *)
          let ctx' = ctx |> Ctx.indent in
          pr_sep "" pr_stmt ss ctx';
          pr_indent ctx';
          pr_expr e1 ctx;
        | _ ->
          pr "match ";
          pr_expr e ctx;
          pr " {";
          pr_list pr_arm arms (ctx |> Ctx.indent);
          pr_indent ctx;
          pr "}";
        end
    | ESpawn (_, _, xs, ts, es) ->
        pr "spawn ";
        pr_path xs ctx;
        pr_type_args ts ctx;
        pr_paren (pr_list pr_expr es) ctx;
  in
  if !Args.show_types || !Args.show_types_stmts then
    begin
      match e with
      | EMatch _ | ELoop _ when not !Args.show_types_stmts ->
          pr_expr e ctx
      | _ ->
          pr_paren (pr_expr e) ctx;
          pr ": ";
          pr_type (Ir1.typeof_expr e) ctx;
    end
  else
    pr_expr e ctx

and pr_arm (p, b) ctx =
  ctx |> pr_indent;
  pr_pat p ctx;
  pr " => ";
  pr_block b ctx;

and pr_pat p ctx =
  let pr_pat p ctx =
    match p with
    | PIgnore (_, _) ->
        pr "_";
    | POr (_, _, p0, p1) ->
        pr_pat p0 ctx;
        pr " or ";
        pr_pat p1 ctx;
    | PRecord (_, _, r) ->
        pr_record_explicit pr_pat r ctx;
    | PConst (_, _, l) ->
        pr_lit l ctx;
    | PVar (_, _, x) ->
        pr_name x ctx;
    | PUnwrap (_, _, x, p) ->
        pr_name x ctx;
        pr_pat p ctx;
  in
  if !Args.show_types then
    begin
      pr_paren (pr_pat p) ctx;
      pr ": ";
      pr_type (Ir1.typeof_pat p) ctx;
    end
  else
    pr_pat p ctx

and pr_path xs ctx =
  if !Args.show_explicit_paths then
    begin
      pr "::";
      Pretty.pr_path xs ctx;
    end
  else
    pr "%s" (last xs);

