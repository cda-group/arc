open Mir
open Pretty
open Utils

let rec pr_mir (mir:Mir.mir) =
  let ctx = Ctx.brief in
  mir |> filter (show_item ctx) |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and show_item (ctx:Ctx.t) (_, i) =
  match i with
  | IExternType | IExternDef _ when not ctx.show_externs -> false
  | _ -> true

and pr_item ((xs, ts), i) ctx =
  ctx |> pr_indent;
  pr "for ";
  pr_brack (pr_types ts) ctx;
  pr " ";
  match i with
  | IVal (t, b) ->
      pr "val ";
      pr_path xs ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b (ctx |> Ctx.indent);
  | IEnum xss ->
      pr "enum ";
      pr_path xs ctx;
      pr " {";
      pr_list pr_path xss (ctx |> Ctx.indent);
      ctx |> pr_indent;
      pr "}";
  | IExternDef (ts, t) ->
      pr "extern def";
      pr_path xs ctx;
      pr_paren (pr_types ts) ctx;
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType ->
      pr "extern type ";
      pr_path xs ctx;
      pr ";";
  | IDef (ps, t, b) ->
      pr "def ";
      pr_path xs ctx;
      pr_params ps ctx;
      pr ": ";
      pr_type t ctx;
      pr " ";
      pr_block b ctx;
  | ITask (ps, i0, i1, b) ->
      pr "task ";
      pr_path xs ctx;
      pr_params ps ctx;
      pr ": ";
      pr_interface i0 ctx;
      pr " -> ";
      pr_interface i1 ctx;
      pr " ";
      pr_block b ctx;
  | IVariant _ -> ()

and pr_interface (xs, ts) ctx =
  pr_path xs ctx;
  pr_brack (pr_types ts) ctx;

and pr_params ps ctx =
  pr_paren (pr_list pr_param ps) ctx

and pr_param (x, t) ctx =
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and path_to_str xs =
  match xs with
  | [] -> ""
  | x::xs -> Printf.sprintf "::%s%s" x (path_to_str xs);

and pr_path xs ctx =
  match xs with
  | [] ->
      ()
  | h::t ->
      pr "::%s" h;
      pr_path t ctx;

and pr_types ts ctx =
  pr_list pr_type ts ctx;

and pr_type t ctx =
  match t with
  | TFunc (ts, t) ->
      pr "fun";
      pr_paren (pr_list pr_type ts) ctx; 
      pr ": ";
      pr_type t ctx;
  | TRecord fts ->
      pr "#";
      pr_brace (pr_list (pr_field pr_type) fts) ctx;
  | TNominal (xs, ts) ->
      pr_path xs ctx;
      if ts != [] then begin
        pr_brack (pr_types ts) ctx;
      end

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
  | EEq (v0, v1) ->
      pr_var v0 ctx;
      pr " == ";
      pr_var v1 ctx;
  | ECall (v, vs) ->
      pr_var v ctx;
      pr_paren (pr_list pr_var vs) ctx;
  | ECast (v, t) ->
      pr_var v ctx;
      pr " as ";
      pr_type t ctx;
  | EEmit v ->
      pr "emit ";
      pr_var v ctx;
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
  | EReceive ->
      pr "receive";
  | ERecord fvs ->
      pr "%%{";
      pr_list (pr_field pr_var) fvs ctx;
      pr "}";
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

and pr_lit l _ctx =
  match l with
  | LInt (c, Some (true, size)) -> pr "%di%d" c size
  | LInt (c, Some (false, size)) -> pr "%du%d" c size
  | LInt (c, None) -> pr "%d" c
  | LFloat (c, Some size) -> pr "%ff%d" c size;
  | LFloat (c, None) -> pr "%f" c;
  | LBool c -> pr "%b" c;
  | LUnit -> pr "unit";
  | LString c -> pr "\"%s\"" c
  | LChar c -> pr "%c" c
