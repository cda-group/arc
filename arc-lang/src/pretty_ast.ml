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

let rec pr_ast (ast:Ast.ast) =
  let ctx = Ctx.brief in
  ast |> filter (show_item ctx) |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and show_item (ctx:Ctx.t) i =
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
  | IVal (x, t, e) ->
      pr "val ";
      pr_name x ctx;
      pr_type_annot t ctx;
      pr " = ";
      pr_expr e ctx;
      pr ";";
  | IEnum (x, gs, xss) ->
      pr "enum ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr " {";
      pr_list pr_variant xss (ctx |> Ctx.indent);
      ctx |> pr_indent;
      pr "}";
  | IExternDef (x, gs, ts, t) ->
      pr "extern fun ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_paren (pr_types ts) ctx;
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType (x, gs) ->
      pr "extern type ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr ";";
  | IDef (x, gs, ps, t0, b) ->
      pr "fun ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr_type_annot t0 ctx;
      pr_body b ctx;
  | ITask (x, gs, ps, i0, i1, b) ->
      pr "task ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr ":";
      pr_interface i0 ctx;
      pr " -> ";
      pr_interface i1 ctx;
      pr_body b ctx;
  | ITypeAlias (x, t) ->
      pr "type ";
      pr_name x ctx;
      pr " = ";
      pr_type t ctx;
      pr ";";
  | IMod (x, is) ->
      pr "mod ";
      pr_name x ctx;
      pr " {";
      pr_sep "" pr_item is ctx;
      pr "}";
  | IUse (xs, x) ->
      pr "use ";
      pr_path xs ctx;
      begin match x with
      | Some x -> pr_name x ctx;
      | None -> ()
      end;
      pr ";";
  | IClass (x, gs, ds) ->
      pr "class ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_decls ds ctx;
  | IInstance (gs, xs, ts, ds) ->
      pr "instance ";
      pr_generics gs ctx;
      pr_path xs ctx;
      pr ": ";
      pr_type_args ts ctx;
      pr_defs ds ctx;

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

and pr_interface i ctx =
  match i with
  | PTagged vs -> pr_list pr_port vs ctx
  | PSingle t -> pr_type t ctx

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
    | EEmit e ->
        pr "emit ";
        pr_expr e ctx;
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
    | EOn arms ->
        pr "on ";
        pr "{";
        pr_sep "," pr_arm arms (ctx |> Ctx.indent);
        ctx |> pr_indent;
        pr "}";
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
        pr_expr e ctx;
    | ETask e ->
        pr "task ";
        pr_expr e ctx;
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

and pr_arm (p, e) ctx =
  ctx |> pr_indent;
  pr_pat p ctx;
  pr " => ";
  pr_expr e ctx;

and pr_binop op _ctx =
  match op with
  | BAdd -> pr "+"
  | BAnd -> pr "and"
  | BBand -> pr "band"
  | BBor -> pr "bor"
  | BBxor -> pr "bxor"
  | BDiv -> pr "/"
  | BEq -> pr "="
  | BGeq -> pr ">="
  | BGt -> pr ">"
  | BLeq -> pr "<="
  | BLt -> pr "<"
  | BMod -> pr "%s" "%"
  | BMul -> pr "*"
  | BMut -> pr "="
  | BNeq -> pr "!="
  | BOr -> pr "|"
  | BPow -> pr "^"
  | BSub -> pr "-"
  | BXor -> pr "xor"
  | BIn -> pr "in"
  | BRExc -> pr ".."
  | BRInc -> pr "..="
  | BBy -> pr "by"
  | BNotIn -> pr "not in"
  | BPipe -> pr "|"

and pr_unop op _ctx =
  match op with
  | UNeg -> pr "-"
  | UNot -> pr "not"

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
