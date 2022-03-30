open Ast
open Utils
open Pretty

let pr_tail f a ctx =
  begin match a with
  | Some a ->
      pr "|";
      f a ctx;
  | None ->
      ()
  end

let rec pr_ast (ast:Ast.ast) debug =
  let ctx = if debug = Debug.Verbose then
    Ctx.verbose
  else
    Ctx.brief
  in
  ast |> filter (show_item ctx) |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and show_item (ctx:'a Ctx.t) i =
  match i with
  | IExternType _ | IExternDef _ when not ctx.show_externs -> false
  | _ -> true

and pr_generics gs ctx =
  if gs != [] then begin
    pr_brack (pr_list pr_generic gs) ctx;
  end;

and pr_generic x ctx =
  pr_name x ctx;

and pr_item i ctx =
  ctx |> pr_indent;
  match i with
  | IVal (d, x, t, e) ->
      pr_decorator d ctx;
      pr "val ";
      pr_name x ctx;
      pr_type_annot t ctx;
      pr " = ";
      pr_expr e ctx;
      pr ";";
  | IEnum (d, x, gs, xss) ->
      pr_decorator d ctx;
      pr "enum ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr " {";
      pr_list pr_variant xss (ctx |> Ctx.indent);
      ctx |> pr_indent;
      pr "}";
  | IExternDef (d, x, gs, ts, t) ->
      pr_decorator d ctx;
      pr "extern fun ";
      pr_def_name x ctx;
      pr_generics gs ctx;
      pr_paren (pr_types ts) ctx;
      pr ": ";
      pr_type_annot t ctx;
      pr ";";
  | IExternType (d, x, gs) ->
      pr_decorator d ctx;
      pr "extern type ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr ";";
  | IDef (d, x, gs, ps, t0, b) ->
      pr_decorator d ctx;
      pr "fun ";
      pr_def_name x ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr_type_annot t0 ctx;
      pr_body b ctx;
  | ITask (d, x, gs, ps0, ps1, b) ->
      pr_decorator d ctx;
      pr "task ";
      pr_def_name x ctx;
      pr_generics gs ctx;
      pr_params ps0 ctx;
      pr ":";
      pr_params ps1 ctx;
      pr_body b ctx;
  | ITypeAlias (a, x, gs, t) ->
      pr_decorator a ctx;
      pr "type ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr " = ";
      pr_type t ctx;
      pr ";";
  | IMod (a, x, is) ->
      pr_decorator a ctx;
      pr "mod ";
      pr_name x ctx;
      pr " {";
      pr_sep "" pr_item is ctx;
      pr "}";
  | IUse (a, xs, x) ->
      pr_decorator a ctx;
      pr "use ";
      pr_path xs ctx;
      begin match x with
      | Some x -> pr_name x ctx;
      | None -> ()
      end;
      pr ";";
  | IClass (d, x, gs, ds) ->
      pr_decorator d ctx;
      pr "class ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_decls ds ctx;
  | IInstance (d, gs, xs, ts, ds) ->
      pr_decorator d ctx;
      pr "instance ";
      pr_generics gs ctx;
      pr_path xs ctx;
      pr ": ";
      pr_type_args ts ctx;
      pr_defs ds ctx;

and pr_def_name d ctx =
  match d with
  | Ast.DName x -> pr_name x ctx;
  | Ast.DUnOp op -> pr_unop op ctx;
  | Ast.DBinOp op -> pr_binop op ctx;

and pr_type_args ts ctx =
  if ts != [] then begin
    pr "[";
    pr_list pr_type ts ctx;
    pr "]";
  end

and pr_decls ds ctx =
  pr " {";
  pr_sep "" pr_decl ds ctx;
  pr "}";

and pr_decl (x, gs, ps, t) ctx =
  pr "def ";
  pr_name x ctx;
  pr_generics gs ctx;
  pr_params ps ctx;
  pr_type_annot t ctx;
  pr ";";

and pr_defs ds ctx =
  if ds != [] then begin
    pr " {";
    pr_sep "" pr_def ds ctx;
    pr "}";
  end;

and pr_def (x, gs, ps, t, b) ctx =
  pr "def ";
  pr_name x ctx;
  pr_generics gs ctx;
  pr_params ps ctx;
  pr_type_annot t ctx;
  pr " ";
  pr_block b ctx;

and pr_body b ctx =
  match b with
  | Some b ->
      pr " ";
      pr_block b ctx
  | None -> pr ";"

and pr_variant (x, ts) ctx =
  ctx |> pr_indent;
  pr_name x ctx;
  match ts with
  | [] -> ()
  | ts ->
    pr_paren (pr_types ts) ctx

and pr_port (x, t) ctx =
  ctx |> pr_indent;
  pr_name x ctx;
  pr_paren (pr_type t) ctx;

and pr_params ps ctx =
  pr_paren (pr_list pr_param ps) ctx;

and pr_param (p, t) ctx =
  pr_pat p ctx;
  pr_type_annot t ctx;

and pr_pat p ctx =
  match p with
  | PIgnore -> pr "_"
  | POr (p0, p1) ->
      pr_pat p0 ctx;
      pr " | ";
      pr_pat p1 ctx;
  | PRecord (fps, p) ->
      pr "#{";
      pr_list (pr_field_opt pr_pat) fps ctx;
      pr_tail pr_pat p ctx;
      pr "}";
  | PTuple ps ->
      pr "(";
      pr_list pr_pat ps ctx;
      pr ",)";
  | PConst l ->
      pr_lit l ctx;
  | PVar x ->
      pr_name x ctx;
  | PUnwrap (xs, ps) ->
      pr_path xs ctx;
      pr_paren (pr_list pr_pat ps) ctx;

and pr_type_annot t ctx =
  match t with
  | Some t ->
      pr ": ";
      pr_type t ctx
  | None -> ()

and pr_types ts ctx =
  pr_list pr_type ts ctx;

and pr_type t ctx =
  match t with
  | TFunc (ts, t) ->
      pr "fun";
      pr_paren (pr_types ts) ctx; 
      pr ": ";
      pr_type t ctx;
  | TTuple ts ->
      pr "(";
      pr_list pr_type ts ctx;
      pr ",)";
  | TRecord (fts, t) ->
      pr "#{";
      pr_list (pr_field_opt pr_type) fts ctx;
      pr_tail pr_type t ctx;
      pr "}";
  | TPath (xs, ts) ->
      pr_type_path xs ts ctx;
  | TArray t ->
      pr_delim "[" "]" (pr_type t) ctx;

and pr_type_path xs ts ctx =
  pr_path xs ctx;
  if ts != [] then begin
    pr_delim "[" "]" (pr_list pr_type ts) ctx; 
  end

and pr_block (ss, e) ctx =
  let ctx' = ctx |> Ctx.indent in
  pr "{";
  begin match (ss, e) with
  | ([], None) -> pr " "
  | ([], Some e) ->
      pr " ";
      pr_expr e ctx';
      pr " ";
  | (ss, Some e) ->
    pr_sep ";" pr_stmt ss ctx';
    pr ";";
    ctx' |> pr_indent;
    pr_expr e ctx';
    ctx |> pr_indent;
  | (ss, None) ->
    pr_sep ";" pr_stmt ss ctx';
    pr ";";
    ctx |> pr_indent
  end;
  pr "}";

and pr_stmt s ctx =
  ctx |> pr_indent;
  match s with
  | SNoop -> ();
  | SVal ((p, t), e) ->
      pr "val ";
      pr_pat p ctx;
      pr_type_annot t ctx;
      pr " = ";
      pr_expr e ctx;
  | SVar ((x, t), e) ->
      pr "var ";
      pr_name x ctx;
      pr_type_annot t ctx;
      pr " = ";
      pr_expr e ctx;
  | SExpr e ->
      pr_expr e ctx;

and pr_expr e ctx =
  let pr_expr e = 
    match e with
    | EWhile (e, b) ->
        pr "while ";
        pr_expr e ctx;
        pr " ";
        pr_block b ctx;
    | EWhileVal (p, e, b) ->
        pr "while val ";
        pr_pat p ctx;
        pr " = ";
        pr_expr e ctx;
        pr " ";
        pr_block b ctx;
    | EAnon ->
        pr "_"
    | EBinOpRef op ->
        pr_paren (pr_binop op) ctx
    | EAccess (e, x) ->
        pr_expr e ctx;
        pr ".";
        pr_name x ctx;
    | EArray (vs, v) ->
        pr "[";
        pr_list pr_expr vs ctx;
        pr_tail pr_expr v ctx;
        pr "]";
    | EBinOp (op, v0, v1) ->
        pr_expr v0 ctx;
        pr " ";
        pr_binop op ctx;
        pr " ";
        pr_expr v1 ctx;
    | ECall (e, vs) ->
        pr_expr e ctx;
        pr_paren (pr_list pr_expr vs) ctx;
    | EInvoke (e, x, vs) ->
        pr_expr e ctx;
        pr ".";
        pr_name x ctx;
        pr_paren (pr_list pr_expr vs) ctx;
    | ECast (e, t) ->
        pr_expr e ctx;
        pr " as ";
        pr_type t ctx;
    | EIf (e, b0, b1) ->
        pr "if ";
        pr_expr e ctx;
        pr " ";
        pr_block b0 ctx;
        begin match b1 with
        | Some b1 -> 
          pr " else ";
          pr_block b1 ctx;
        | None -> ()
        end
    | EIfVal (p, e, b0, b1) ->
        pr "if let ";
        pr_pat p ctx;
        pr " = ";
        pr_expr e ctx;
        pr " ";
        pr_block b0 ctx;
        begin match b1 with
        | Some b1 ->
            pr " else ";
            pr_block b1 ctx;
        | None -> ()
        end
    | ELit l ->
        pr_lit l ctx;
    | ELoop b ->
        pr "loop ";
        pr_block b ctx;
    | EOn receivers ->
        pr "on ";
        pr "{";
        pr_sep "," pr_receiver receivers (ctx |> Ctx.indent);
        ctx |> pr_indent;
        pr "}";
    | EReceive e ->
        pr "receive ";
        pr_expr e ctx;
    | EEmit (e0, e1) ->
        pr_expr e0 ctx;
        pr "!";
        pr_expr e1 ctx;
    | ESelect (e0, e1) ->
        pr_expr e0 ctx;
        pr "[";
        pr_expr e1 ctx;
        pr "]";
    | ERecord (fvs, v) ->
        pr "#{";
        pr_list (pr_field_opt pr_expr) fvs ctx;
        pr_tail pr_expr v ctx;
        pr "}";
    | EUnOp (op, e) ->
        pr_unop op ctx;
        pr_expr e ctx;
    | EReturn e ->
        begin match e with
        | Some e ->
            pr "return ";
            pr_expr e ctx;
        | None ->
            pr "return"
        end
    | EBreak e ->
        begin match e with
        | Some e ->
            pr "break ";
            pr_expr e ctx;
        | None ->
            pr "break"
        end
    | EContinue ->
        pr "continue"
    | ETuple es ->
        pr "(";
        pr_list pr_expr es ctx;
        pr ",)";
    | EProject (e, i) ->
        pr_expr e ctx;
        pr ".%d" i;
    | EBlock (b) ->
        pr_block b ctx;
    | EFunc (ps, e) ->
        pr "fun";
        pr_paren (pr_list pr_param ps) ctx;
        pr ": ";
        pr_block e ctx;
    | ETask (ps, e) ->
        pr "task: ";
        pr_params ps ctx;
        pr_block e ctx;
    | EFor (p, e, b) ->
        pr "for ";
        pr_pat p ctx;
        pr " in ";
        pr_expr e ctx;
        pr_block b ctx;
    | EMatch (e, arms) ->
        pr "match ";
        pr_expr e ctx;
        pr " {";
        pr_sep "," pr_arm arms (ctx |> Ctx.indent);
        ctx |> pr_indent;
        pr "}";
    | ECompr (e0, (p, e), cs) ->
        pr "[";
        pr_expr e0 ctx;
        pr " ";
        pr "on ";
        pr_pat p ctx;
        pr " in ";
        pr_expr e ctx;
        pr_sep " " pr_clause cs ctx;
        pr "]";
    | EPath (xs, ts) ->
        pr_path xs ctx;
        if ts != [] then begin
          pr "::";
          pr_brack (pr_list pr_type ts) ctx
        end
    | EFrom _ -> todo ()
  in
  if ctx.show_types then begin
    pr_paren pr_expr e;
  end else
    pr_expr e

and pr_clause c ctx =
  match c with
  | CFor (p, e) ->
      pr " for ";
      pr_pat p ctx;
      pr " in ";
      pr_expr e ctx;
  | CIf e ->
      pr " if ";
      pr_expr e ctx;

and pr_receiver (p, e0, e1) ctx =
  pr_pat p ctx;
  pr " in ";
  pr_expr e0 ctx;
  pr " => ";
  pr_expr e1 ctx

and pr_arm (p, e) ctx =
  ctx |> pr_indent;
  pr_pat p ctx;
  pr " => ";
  pr_expr e ctx;

and pr_suffixed x s _ctx =
  match s with
  | Some s ->
      pr "%s%s" x s
  | None ->
      pr "%s" x

and pr_binop op _ctx =
  match op with
  | BAdd s -> pr_suffixed "+" s _ctx
  | BAnd -> pr "and"
  | BBand -> pr "band"
  | BBor -> pr "bor"
  | BBxor -> pr "bxor"
  | BDiv s -> pr_suffixed "/" s _ctx
  | BEq s -> pr_suffixed "==" s _ctx
  | BGeq s -> pr_suffixed ">=" s _ctx
  | BGt s -> pr_suffixed ">" s _ctx
  | BLeq s -> pr_suffixed "<=" s _ctx
  | BLt s -> pr_suffixed "<" s _ctx
  | BMod s -> pr_suffixed "%%" s _ctx
  | BMul s -> pr_suffixed "*" s _ctx
  | BMut -> pr "="
  | BNeq s -> pr_suffixed "!=" s _ctx
  | BOr -> pr "|"
  | BPow s -> pr_suffixed "**" s _ctx
  | BSub s -> pr_suffixed "-" s _ctx
  | BXor -> pr "xor"
  | BIn -> pr "in"
  | BRExc -> pr ".."
  | BRInc -> pr "..="
  | BBy -> pr "by"
  | BNotIn -> pr "not in"

and pr_unop op _ctx =
  match op with
  | UNeg s -> pr_suffixed "-" s _ctx
  | UNot -> pr "not"
