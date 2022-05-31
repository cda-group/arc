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
  let ctx = Ctx.make in
  pr_items ast ctx;
  pr "\n";

and show_item i =
  match Ast.item_loc i with
  | LocStd _ | NoLocStd when not !Args.show_std -> false
  | _ ->
    match i with
    | IExternType _ | IExternDef _ when not !Args.show_externs -> false
    | _ -> true

and pr_generics gs ctx =
  if gs != [] then begin
    pr_brack (pr_list pr_generic gs) ctx;
  end;

and pr_generic x ctx =
  pr_name x ctx;

and pr_items is ctx =
  is |> filter show_item |> List.iter (fun i -> pr_item i ctx);

and pr_item i ctx =
  ctx |> pr_indent;
  match i with
  | IVal (_, d, x, t, e) ->
      pr_decorator d ctx;
      pr "val ";
      pr_name x ctx;
      pr_type_annot t ctx;
      pr " = ";
      pr_expr e ctx;
      pr ";";
  | IExternDef (_, d, async, x, gs, ts, t, bs) ->
      pr_decorator d ctx;
      pr "extern";
      if async then begin
        pr " async";
      end;
      pr " def ";
      pr_def_name x ctx;
      pr_generics gs ctx;
      pr_paren (pr_types ts) ctx;
      pr_where bs ctx;
      pr_type_annot t ctx;
      pr ";";
  | IExternType (_, d, x, gs, bs) ->
      pr_decorator d ctx;
      pr "extern type ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_where bs ctx;
      pr ";";
  | IDef (_, d, async, x, gs, ps, t0, bs, b) ->
      pr_decorator d ctx;
      if async then begin
        pr " async";
      end;
      pr "def ";
      pr_def_name x ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr_type_annot t0 ctx;
      pr_where bs ctx;
      pr_body b ctx;
  | ITask (_, d, x, gs, pts0, xts1, bs, b) ->
      pr_decorator d ctx;
      pr "task ";
      pr_def_name x ctx;
      pr_generics gs ctx;
      pr_params pts0 ctx;
      pr ": ";
      pr_paren (pr_fields_opt pr_type xts1) ctx;
      pr_where bs ctx;
      pr_body b ctx;
  | IType (_, a, x, gs, t, bs) ->
      pr_decorator a ctx;
      pr "type ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr " = ";
      pr_type t ctx;
      pr_where bs ctx;
      pr ";";
  | IMod (_, a, x, is) ->
      pr_decorator a ctx;
      pr "mod ";
      pr_name x ctx;
      pr " {";
      let ctx' = ctx |> Ctx.indent in
      pr_items is ctx';
      pr_indent ctx;
      pr "}";
  | IUse (_, a, xs, s) ->
      pr_decorator a ctx;
      pr "use ";
      pr_path xs ctx;
      begin match s with
      | Some UAlias x ->
          pr " as ";
          pr_name x ctx;
      | Some UGlob ->
          pr "*";
      | None -> ()
      end;
      pr ";";
  | IClass (_, d, x, gs, bs, ds) ->
      pr_decorator d ctx;
      pr "class ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_where bs ctx;
      pr " {";
      pr_sep "" pr_decl ds (ctx |> Ctx.indent);
      pr_indent ctx;
      pr "}";
  | IInstance (_, d, gs, xs, ts, bs, ds) ->
      pr_decorator d ctx;
      pr "instance ";
      pr_generics gs ctx;
      pr_path xs ctx;
      pr_type_args ts ctx;
      pr_where bs ctx;
      pr " {";
      pr_sep "" pr_def ds (ctx |> Ctx.indent);
      pr_indent ctx;
      pr "}";

and pr_where bs ctx =
  if bs != [] then begin
    pr " where ";
    pr_list pr_bound bs ctx;
  end;

and pr_bound (xs, ts) ctx =
  pr_path xs ctx;
  pr_type_args ts ctx;

and pr_def_name d ctx =
  match d with
  | Ast.DName x -> pr_name x ctx;
  | Ast.DUnOp (op, _) -> pr_unop op ctx;
  | Ast.DBinOp (op, _) -> pr_binop op ctx;

and pr_type_args ts ctx =
  if ts <> [] then begin
    pr_brack (pr_list pr_type ts) ctx;
  end

and pr_explicit_type_args ts ctx =
  if ts <> [] then begin
    pr "::";
    pr_brack (pr_list pr_type ts) ctx;
  end

and pr_decl (x, gs, ps, t, bs) ctx =
  pr_indent ctx;
  pr "def ";
  pr_name x ctx;
  pr_generics gs ctx;
  pr_params ps ctx;
  pr_type_annot t ctx;
  pr_where bs ctx;
  pr ";";

and pr_def (x, gs, ps, t, bs, b) ctx =
  pr_indent ctx;
  pr "def ";
  pr_name x ctx;
  pr_generics gs ctx;
  pr_params ps ctx;
  pr_type_annot t ctx;
  pr_where bs ctx;
  pr " ";
  pr_block b ctx;

and pr_body b ctx =
  match b with
  | ([], Some e) ->
      pr " = ";
      pr_expr e ctx;
  | b ->
      pr " ";
      pr_block b ctx

and pr_variant (_, x, ts) ctx =
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
  | PIgnore _ -> pr "_"
  | POr (_, p0, p1) ->
      pr_pat p0 ctx;
      pr " | ";
      pr_pat p1 ctx;
  | PRecord (_, rp) ->
      pr_record pr_pat rp ctx;
  | PTuple (_, ps) ->
      pr "(";
      pr_list pr_pat ps ctx;
      pr ",)";
  | PArray (_, ps, p) ->
      pr "[";
      pr_list pr_pat ps ctx;
      pr_tail pr_pat p ctx;
      pr "]";
  | PConst (_, l) ->
      pr_lit l ctx;
  | PVar (_, x) ->
      pr_name x ctx;
  | PUnwrap (_, x, ps) ->
      pr_name x ctx;
      pr_pat ps ctx;

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
  | TFunc (_, ts, t) ->
      pr "fun";
      pr_paren (pr_types ts) ctx; 
      pr ": ";
      pr_type t ctx;
  | TEnum (_, et) ->
      pr "enum ";
      pr_enum pr_type et ctx;
  | TTuple (_, ts) ->
      pr "(";
      pr_list pr_type ts ctx;
      pr ",)";
  | TRecord (_, rt) ->
      pr_record pr_type rt ctx;
  | TPath (_, xs, ts) ->
      pr_type_path xs ts ctx;
  | TArray (_, t) ->
      pr_delim "[" "]" (pr_type t) ctx;

and pr_type_path xs ts ctx =
  pr_path xs ctx;
  if ts != [] then begin
    pr_brack  (pr_list pr_type ts) ctx; 
  end

and pr_block (ss, e) ctx =
  let ctx' = ctx |> Ctx.indent in
  pr "{";
  begin match (ss, e) with
  | ([], None) -> pr " "
  | ([], Some e) ->
      pr_indent ctx';
      pr_expr e ctx';
      pr_indent ctx;
  | (ss, Some e) ->
    pr_sep ";" pr_stmt ss ctx';
    pr ";";
    pr_indent ctx';
    pr_expr e ctx';
    pr_indent ctx;
  | (ss, None) ->
    pr_sep ";" pr_stmt ss ctx';
    pr ";";
    ctx |> pr_indent
  end;
  pr "}";

and pr_stmt s ctx =
  ctx |> pr_indent;
  match s with
  | SNoop _ -> ();
  | SVal (_, (p, t), e) ->
      pr "val ";
      pr_pat p ctx;
      pr_type_annot t ctx;
      pr " = ";
      pr_expr e ctx;
  | SVar (_, (x, t), e) ->
      pr "var ";
      pr_name x ctx;
      pr_type_annot t ctx;
      pr " = ";
      pr_expr e ctx;
  | SExpr (_, e) ->
      pr_expr e ctx;

and pr_expr e ctx =
  let pr_expr e = 
    match e with
    | EWhile (_, e, b) ->
        pr "while ";
        pr_expr e ctx;
        pr " ";
        pr_block b ctx;
    | EWhileVal (_, p, e, b) ->
        pr "while val ";
        pr_pat p ctx;
        pr " = ";
        pr_expr e ctx;
        pr " ";
        pr_block b ctx;
    | EAnon _ ->
        pr "_"
    | EBinOpRef (_, op) ->
        pr_paren (pr_binop op) ctx
    | EAccess (_, e, x) ->
        pr_expr e ctx;
        pr ".";
        pr_name x ctx;
    | EArray (_, es, e) ->
        pr "[";
        pr_list pr_expr es ctx;
        pr_tail pr_expr e ctx;
        pr "]";
    | EBinOp (_, op, ts, e0, e1) ->
        pr_expr e0 ctx;
        pr " ";
        pr_binop op ctx;
        if ts <> [] then begin
          pr "::";
          pr_brack (pr_types ts) ctx
        end;
        pr " ";
        pr_expr e1 ctx;
    | ECall (_, e, es) ->
        pr_expr e ctx;
        pr_paren (pr_list pr_expr es) ctx;
    | EInvoke (_, e, x, es) ->
        pr_expr e ctx;
        pr ".";
        pr_name x ctx;
        pr_paren (pr_list pr_expr es) ctx;
    | ECast (_, e, t) ->
        pr_expr e ctx;
        pr " as ";
        pr_type t ctx;
    | EIf (_, e, b0, b1) ->
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
    | EIfVal (_, p, e, b0, b1) ->
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
    | ELit (_, l) ->
        pr_lit l ctx;
    | ELoop (_, b) ->
        pr "loop ";
        pr_block b ctx;
    | EOn (_, receivers) ->
        pr "on ";
        pr "{";
        pr_sep "," pr_receiver receivers (ctx |> Ctx.indent);
        ctx |> pr_indent;
        pr "}";
    | ESelect (_, e0, e1) ->
        pr_expr e0 ctx;
        pr "[";
        pr_expr e1 ctx;
        pr "]";
    | ERecord (_, re) ->
        pr_record pr_expr re ctx;
    | EEnwrap (_, x, e) ->
        pr_name x ctx;
        pr_expr e ctx;
    | EUnOp (_, op, ts, e) ->
        pr_unop op ctx;
        pr_expr e ctx;
        if ts <> [] then begin
          pr "::";
          pr_brack (pr_types ts) ctx
        end
    | EReturn (_, e) ->
        begin match e with
        | Some e ->
            pr "return ";
            pr_expr e ctx;
        | None ->
            pr "return"
        end
    | EBreak (_, e) ->
        begin match e with
        | Some e ->
            pr "break ";
            pr_expr e ctx;
        | None ->
            pr "break"
        end
    | EContinue _ ->
        pr "continue"
    | ETuple (_, es) ->
        pr "(";
        pr_list pr_expr es ctx;
        pr ",)";
    | EProject (_, e, i) ->
        pr_expr e ctx;
        pr_index i ctx;
    | EBlock (_, b) ->
        pr_block b ctx;
    | EFunc (_, ps, b) ->
        pr "fun";
        pr_paren (pr_list pr_param ps) ctx;
        pr_body b ctx;
    | ETask (_, ps, xts, e) ->
        pr "task";
        pr_params ps ctx;
        pr ": ";
        pr_paren (pr_fields_opt pr_type xts) ctx;
        pr_block e ctx;
    | EFor (_, p, e, b) ->
        pr "for ";
        pr_pat p ctx;
        pr " in ";
        pr_expr e ctx;
        pr_block b ctx;
    | EMatch (_, e, arms) ->
        pr "match ";
        pr_expr e ctx;
        pr " {";
        pr_sep "," pr_arm arms (ctx |> Ctx.indent);
        ctx |> pr_indent;
        pr "}";
    | EPath (_, xs, ts) ->
        pr_path xs ctx;
        if ts != [] then begin
          pr "::";
          pr_brack (pr_list pr_type ts) ctx
        end
    | EFrom _ -> todo ()
    | ETry (_, b0, arms, b1) ->
        pr "try ";
        pr_block b0 ctx;
        pr " catch ";
        pr_sep "," pr_arm arms (ctx |> Ctx.indent);
        begin match b1 with
        | Some b1 ->
            pr " finally ";
            pr_block b1 ctx;
        | None -> ()
        end;
    | EThrow (_, e) ->
        pr "throw ";
        pr_expr e ctx;
  in
  if !Args.verbose then
    pr_paren pr_expr e
  else
    pr_expr e

and pr_index i _ctx =
  pr ".%d" i

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

and pr_binop op _ctx =
  match op with
  | BAdd -> pr "+"
  | BAnd -> pr "and"
  | BBand -> pr "band"
  | BBor -> pr "bor"
  | BBxor -> pr "bxor"
  | BDiv -> pr "/"
  | BEq -> pr "=="
  | BGeq -> pr ">="
  | BGt -> pr ">"
  | BLeq -> pr "<="
  | BLt -> pr "<"
  | BMod -> pr "%%"
  | BMul -> pr "*"
  | BMut -> pr "="
  | BNeq -> pr "!="
  | BOr -> pr "|"
  | BPow -> pr "**"
  | BSub -> pr "-"
  | BXor -> pr "xor"
  | BIn -> pr "in"
  | BRExc -> pr ".."
  | BRInc -> pr "..="
  | BBy -> pr "by"
  | BNotIn -> pr "not in"

and pr_unop op _ctx =
  match op with
  | UNeg -> pr "-"
  | UNot -> pr "not "

and pr_path p ctx =
  match p with
  | PAbs xs ->
      pr "::";
      Pretty.pr_path xs ctx
  | PRel xs ->
      Pretty.pr_path xs ctx

and path_to_str p =
  match p with
  | PAbs xs ->
      "::" ^ (Pretty.path_to_str xs)
  | PRel xs ->
      Pretty.path_to_str xs
