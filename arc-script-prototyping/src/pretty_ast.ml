open Ast

module Ctx = struct
  type t = {
    indent: int;
    debug: bool;
    ast: Ast.ast;
  }

  let print_indent ctx: unit =
    Printf.printf "\n";
    let rec print_indent = function
      | 0 -> ()
      | i ->
        Printf.printf "    ";
        print_indent (i - 1)
    in
    print_indent ctx.indent

  let indent ctx =
    { ctx with indent = ctx.indent + 1 }

  and make ast debug = {
      indent = 0;
      debug;
      ast;
    }

end

let pr fmt = Printf.printf fmt

let rec pr_sep sep f l (ctx:Ctx.t) =
  match l with
  | [x]  -> f x ctx
  | []   -> ()
  | h::t ->
      f h ctx;
      pr sep;
      pr_sep sep f t ctx

let rec pr_ast (ast:Ast.ast) =
  let ctx = Ctx.make ast false in
  ast |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and pr_generics gs ctx =
  if gs != [] then begin
    pr "[";
    pr_sep ", " pr_generic gs ctx;
    pr "]";
  end;

and pr_generic x ctx =
  pr_name x ctx;

and pr_item i ctx =
  ctx |> Ctx.print_indent;
  match i with
  | IVal (x, t, e) ->
      pr "val ";
      pr_name x ctx;
      pr_type_opt t ctx;
      pr " = ";
      pr_expr e ctx;
      pr ";";
  | IEnum (x, gs, xss) ->
      pr "enum ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr " {";
      pr_sep ", " pr_variant xss (ctx |> Ctx.indent);
      ctx |> Ctx.print_indent;
      pr "}";
  | IExternFunc (x, gs, ps, t) ->
      pr "extern fun ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr "(";
      pr_sep ", " pr_basic_param ps ctx;
      pr ")";
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType (x, gs) ->
      pr "extern type ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr ";";
  | IFunc (x, gs, ps, t, b) ->
      pr "fun ";
      pr_name x ctx;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr_type_opt t ctx;
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

and pr_body b ctx =
  match b with
  | Some b ->
      pr " ";
      pr_block b ctx
  | None -> pr ";"

and pr_interface i ctx =
  match i with
  | PTagged vs -> pr_sep ", " pr_port vs ctx
  | PSingle t -> pr_type t ctx

and pr_variant (x, ts) ctx =
  ctx |> Ctx.print_indent;
  pr_name x ctx;
  match ts with
  | [] -> ()
  | ts ->
    pr "(";
    pr_sep ", " pr_type ts ctx;
    pr ")";

and pr_port (x, t) ctx =
  ctx |> Ctx.print_indent;
  pr_name x ctx;
  pr "(";
  pr_type t ctx;
  pr ")";

and pr_basic_params ps ctx =
  pr_sep ", " pr_basic_param ps ctx;

and pr_basic_param (x, t) ctx =
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_params ps ctx =
  pr "(";
  pr_sep ", " pr_param ps ctx;
  pr ")";

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
  | PRecord fps ->
      pr "%%{";
      pr_sep ", " pr_field_pat fps ctx;
      pr "}";
  | PTuple ps ->
      pr "(";
      pr_sep ", " pr_pat ps ctx;
      pr ",)";
  | PConst l ->
      pr_lit l ctx;
  | PVar x ->
      pr_name x ctx;
  | PUnwrap (xs, ps) ->
      pr_path xs ctx;
      pr "(";
      pr_sep ", " pr_pat ps ctx;
      pr ")";

and pr_field_pat (x, p) ctx =
  pr_name x ctx;
  match p with
  | Some p ->
      pr ": ";
      pr_pat p ctx;
  | None -> ()

and pr_path xs ctx =
  match xs with
  | [] ->
      ()
  | [h] ->
      pr "%s" h;
  | h::t ->
      pr "%s::" h;
      pr_path t ctx;

and pr_type_annot t ctx =
  match t with
  | Some t ->
      pr ": ";
      pr_type t ctx
  | None -> ()

and pr_type_opt t ctx =
  match t with
  | Some t ->
      pr ": ";
      pr_type t ctx
  | None -> ()

and pr_types ts ctx =
  pr "(";
  pr_sep ", " pr_type ts ctx;
  pr ")";

and pr_type t ctx =
  match t with
  | TFunc (ts, t) ->
      pr "fun";
      pr_types ts ctx; 
      pr ": ";
      pr_type t ctx;
  | TTuple ts ->
      pr "(";
      pr_sep ", " pr_type ts ctx;
      pr ",)";
  | TRecord (fts, t) ->
      pr "%%{ ";
      pr_sep ", " pr_field_type fts ctx;
      begin match t with
      | Some t ->
          pr "|";
          pr_type t ctx;
      | None ->
          ()
      end;
      pr " }";
  | TPath (xs, ts) ->
      pr_path xs ctx;
      if ts != [] then begin
        pr "[";
        pr_sep ", " pr_type ts ctx; 
        pr "]";
      end
  | TArray t ->
      pr "[";
      pr_type t ctx;
      pr "]";
  | TStream t ->
      pr "~";
      pr_type t ctx;

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
    ctx' |> Ctx.print_indent;
    pr_expr e ctx';
    ctx |> Ctx.print_indent;
  | (ss, None) ->
    pr_sep ";" pr_stmt ss ctx';
    pr ";";
    ctx |> Ctx.print_indent
  end;
  pr "}";

and pr_stmt s ctx =
  ctx |> Ctx.print_indent;
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

and pr_name x _ctx =
  pr "%s" x;

and pr_expr e ctx =
  let pr_expr e = 
    match e with
    | EAccess (e, x) ->
        pr_expr e ctx;
        pr ".";
        pr_name x ctx;
    | EAfter (e, b) ->
        pr "after ";
        pr_expr e ctx;
        pr " ";
        pr_block b (ctx |> Ctx.indent);
    | EEvery (e, b) ->
        pr "every ";
        pr_expr e ctx;
        pr " ";
        pr_block b (ctx |> Ctx.indent);
    | EArray vs ->
        pr "[";
        pr_sep ", " pr_expr vs ctx;
        pr "]";
    | EBinOp (op, v0, v1) ->
        pr_expr v0 ctx;
        pr " ";
        pr_binop op ctx;
        pr " ";
        pr_expr v1 ctx;
    | ECall (e, vs) ->
        pr_expr e ctx;
        pr "(";
        pr_sep ", " pr_expr vs ctx;
        pr ")";
    | EInvoke (e, x, vs) ->
        pr_expr e ctx;
        pr ".";
        pr_name x ctx;
        pr "(";
        pr_sep ", " pr_expr vs ctx;
        pr ")";
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
    | ELog e ->
        pr "log ";
        pr_expr e ctx;
    | ELoop b ->
        pr "loop ";
        pr_block b ctx;
    | EOn arms ->
        pr "on ";
        pr "{";
        pr_sep "," pr_arm arms (ctx |> Ctx.indent);
        ctx |> Ctx.print_indent;
        pr "}";
    | ESelect (e0, e1) ->
        pr_expr e0 ctx;
        pr "[";
        pr_expr e1 ctx;
        pr "]";
    | ERecord fvs ->
        pr "%%{";
        pr_sep ", " pr_field_expr fvs ctx;
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
        pr_sep ", " pr_expr es ctx;
        pr ",)";
    | EProject (e, i) ->
        pr_expr e ctx;
        pr ".%d" i;
    | EBlock (b) ->
        pr_block b ctx;
    | EFunc (ps, e) ->
        pr "fun(";
        pr_sep ", " pr_param ps ctx;
        pr "): ";
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
        ctx |> Ctx.print_indent;
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
          pr "::[";
          pr_sep ", " pr_type ts ctx;
          pr "]";
        end
    | EWith (e, es) ->
        pr_expr e ctx;
        pr " with {";
        pr_sep ", " pr_field_expr es ctx;
        pr "}";
  in
  if ctx.debug then begin
    pr "(";
    pr_expr e;
    pr ")";
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
  ctx |> Ctx.print_indent;
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

and pr_field_type (x, t) ctx =
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_field_expr (x, e) ctx =
  pr_name x ctx;
  match e with
  | Some e -> 
    pr ": ";
    pr_expr e ctx;
  | None -> ()

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