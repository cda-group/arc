open Hir

module Ctx = struct
  type t = {
    indent: int;
    debug: bool;
    hir: Hir.hir;
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

  and make hir debug = {
      indent = 0;
      debug;
      hir;
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

let rec pr_hir (hir:Hir.hir) =
  let ctx = Ctx.make hir false in
  hir |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

and pr_item (xs, i) ctx =
  ctx |> Ctx.print_indent;
  match i with
  | IVal (t, b) ->
      pr "val ";
      pr_path xs;
      pr ": ";
      pr_type t ctx;
      pr " = ";
      pr_block b (ctx |> Ctx.indent);
  | IEnum (gs, xss) ->
      pr "enum ";
      pr_path xs;
      pr_generics gs ctx;
      pr " {";
      pr_sep ", " pr_variant_path xss (ctx |> Ctx.indent);
      ctx |> Ctx.print_indent;
      pr "}";
  | IExternFunc (gs, ps, t) ->
      pr "extern fun";
      pr_path xs;
      pr_generics gs ctx;
      pr "(";
      pr_sep ", " pr_param ps ctx;
      pr ")";
      pr ": ";
      pr_type t ctx;
      pr ";";
  | IExternType gs ->
      pr "extern type ";
      pr_path xs;
      pr_generics gs ctx;
      pr ";";
  | IFunc (gs, ps, t, b) ->
      pr "fun ";
      pr_path xs;
      pr_generics gs ctx;
      pr_params ps ctx;
      pr ": ";
      pr_type t ctx;
      pr " ";
      pr_block b ctx;
  | ITask (gs, ps, i0, i1, b) ->
      pr "task ";
      pr_path xs;
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
      pr_path xs;
      pr_generics gs ctx;
      pr " = ";
      pr_type t ctx;
      pr ";";
  | IVariant _ -> ()

and pr_interface (xs, ts) ctx =
  pr_path xs;
  pr "[";
  pr_sep ", " pr_type ts ctx;
  pr "]";

and pr_generics gs ctx =
  if gs != [] then begin
    pr "[";
    pr_sep ", " pr_generic gs ctx;
    pr "]";
  end

and pr_generic g ctx =
  pr_name g ctx;

and pr_params ps ctx =
  pr "(";
  pr_sep ", " pr_param ps ctx;
  pr ")";

and pr_param (x, t) ctx =
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_variant_path xs (ctx:Ctx.t) =
  match ctx.hir |> List.assoc xs with
  | IVariant t ->
      ctx |> Ctx.print_indent;
      pr_path xs;
      pr "(";
      pr_type t ctx;
      pr ")";
  | _ -> assert false

and path_to_str xs =
  match xs with
  | [] -> ""
  | x::xs -> Printf.sprintf "::%s%s" x (path_to_str xs);

and pr_path xs =
  match xs with
  | [] ->
      ()
  | h::t ->
      pr "::%s" h;
      pr_path t;

and debug_type t =
  pr "\nDEBUG: ";
  pr_type t (Ctx.make [] false)

and pr_type t ctx =
  match t with
  | TFunc (ts, t) ->
      pr "fun(";
      pr_sep ", " pr_type ts ctx; 
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
      pr_path xs;
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
  | TVar x -> pr "'%s" x;
  | TGeneric x -> pr "%s" x;

and pr_block (ss, v) ctx =
  let ctx' = ctx |> Ctx.indent in
  pr "{";
  if ss != [] then begin
    pr_sep ";" pr_ssa ss ctx';
    pr ";";
  end;
  ctx' |> Ctx.print_indent;
  pr_var v ctx';
  ctx |> Ctx.print_indent;
  pr "}";

and pr_ssa (x, t, e) ctx =
  ctx |> Ctx.print_indent;
  pr "val ";
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;
  pr " = ";
  pr_expr e ctx;

and debug_name x =
  pr_name x (Ctx.make [] false)

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
  | EArray vs ->
      pr "[";
      pr_sep ", " pr_var vs ctx;
      pr "]";
  | EBinOp (op, v0, v1) ->
      pr_var v0 ctx;
      pr " ";
      pr_binop op ctx;
      pr " ";
      pr_var v1 ctx;
  | ECall (v, vs) ->
      pr_var v ctx;
      pr "(";
      pr_sep ", " pr_var vs ctx;
      pr ")";
  | ECast (v, t) ->
      pr_var v ctx;
      pr " as ";
      pr_type t ctx;
  | EEmit v ->
      pr "emit ";
      pr_var v ctx;
  | EEnwrap (xs, ts, v) ->
      pr "enwrap[";
      pr_path xs;
      if ts != [] then begin
        pr "[";
        pr_sep ", " pr_type ts ctx;
        pr "]";
      end;
      pr "]";
      pr "(";
      pr_var v ctx;
      pr ")";
  | EIf (v, b0, b1) ->
      pr "if ";
      pr_var v ctx;
      pr " ";
      pr_block b0 ctx;
      pr " else ";
      pr_block b1 ctx;
  | EIs (xs, ts, v) ->
      pr "is[";
      pr_path xs;
      if ts != [] then begin
        pr "[";
        pr_sep ", " pr_type ts ctx;
        pr "]";
      end;
      pr "]";
      pr "(";
      pr_var v ctx;
      pr ")";
  | ELit l ->
      pr_lit l ctx;
  | ELog v ->
      pr "log ";
      pr_var v ctx;
  | ELoop b ->
      pr "loop ";
      pr_block b ctx;
  | EReceive ->
      pr "receive";
  | ESelect (v0, v1) ->
      pr_var v0 ctx;
      pr "[";
      pr_var v1 ctx;
      pr "]";
  | ERecord fvs ->
      pr "%%{";
      pr_sep ", " pr_field_expr fvs ctx;
      pr "}";
  | EUnOp (op, v) ->
      pr_unop op ctx;
      pr_var v ctx;
  | EUnwrap (xs, ts, v) ->
      pr "unwrap[";
      pr_path xs;
      if ts != [] then begin
        pr "[";
        pr_sep ", " pr_type ts ctx;
        pr "]";
      end;
      pr "]";
      pr "(";
      pr_var v ctx;
      pr ")";
  | EReturn v ->
      pr "return ";
      pr_var v ctx;
  | EBreak v ->
      pr "break ";
      pr_var v ctx;
  | EContinue ->
      pr "continue"
  | EItem (xs, ts) ->
      pr_path xs;
      if ts != [] then begin
        pr "[";
        pr_sep ", " pr_type ts ctx;
        pr "]";
      end

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
  | BMod -> pr "%s" "%"
  | BMul -> pr "*"
  | BNeq -> pr "!="
  | BOr -> pr "|"
  | BPow -> pr "^"
  | BSub -> pr "-"
  | BXor -> pr "xor"
  | BIn -> pr "in"
  | BRExc -> pr ".."
  | BRInc -> pr "..="
  | BWith -> pr "with"

and pr_unop op _ctx =
  match op with
  | UNeg -> pr "-"
  | UNot -> pr "not"

and pr_var x ctx =
  pr_name x ctx

and pr_field_type (x, t) ctx =
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_field_expr (x, v) ctx =
  pr_name x ctx;
  pr ": ";
  pr_var v ctx;

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
