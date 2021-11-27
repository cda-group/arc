open Hir
open Pretty

let rec pr_hir (hir:Hir.hir) =
  let ctx = Ctx.brief in
  hir |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and pr_item (xs, i) ctx =
  ctx |> pr_indent;
  match i with
  | IVal (t, b) ->
      pr "val ";
      pr_path xs ctx;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b (ctx |> Ctx.indent);
  | IEnum (gs, xss) ->
      pr "enum ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr " {";
      pr_list pr_path xss (ctx |> Ctx.indent);
      ctx |> pr_indent;
      pr "}";
  | IExternDef (gs, ts, t) ->
      pr "extern fun";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_paren (pr_types ts) ctx;
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType gs ->
      pr "extern type ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr ";";
  | IDef (gs, ps, t, b) ->
      pr "def ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr ": ";
      pr_type t ctx;
      pr " ";
      pr_block b ctx;
  | ITask (gs, ps, i0, i1, b) ->
      pr "task ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr ": ";
      pr_interface i0 ctx;
      pr " -> ";
      pr_interface i1 ctx;
      pr " ";
      pr_block b ctx;
  | ITypeAlias (gs, t) ->
      pr "type ";
      pr_path xs ctx;
      pr_generics gs ctx;
      pr " = ";
      pr_type t ctx;
      pr ";";
  | IVariant _ -> ()
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

and debug_type t =
  pr "\nDEBUG: ";
  pr_type t Ctx.brief

and pr_types ts ctx =
  pr_list pr_type ts ctx;

and pr_type t ctx =
  match t with
  | TFunc (ts, t) ->
      pr "fun(";
      pr_list pr_type ts ctx; 
      pr "): ";
      pr_type t ctx;
  | TRecord t ->
      pr "%%{";
      pr_type t ctx;
      pr "}";
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

and debug_name x =
  pr_name x Ctx.brief

and pr_name x _ctx =
  pr "%s" x;

and pr_expr e ctx =
  match e with
  | EAccess (v, x) ->
      pr_var v ctx;
      pr ".";
      pr_name x ctx;
  | EAfter (v, b) ->
      pr "after ";
      pr_var v ctx;
      pr " ";
      pr_block b (ctx |> Ctx.indent);
  | EEvery (v, b) ->
      pr "every ";
      pr_var v ctx;
      pr " ";
      pr_block b (ctx |> Ctx.indent);
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
