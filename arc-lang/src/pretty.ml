
module Ctx = struct
  type 'a t = {
    indent: int;
    show_types: bool;
    show_externs: bool;
    show_parens: bool;
    data: 'a;
  }

  let indent ctx =
    { ctx with indent = ctx.indent + 1 }

  let make show_types show_externs show_parens data = {
      indent = 0;
      show_types;
      show_externs;
      show_parens;
      data;
    }

  let brief = make false false false ()
  let typed = make true false false ()
  let verbose = make true true true ()

  let brief_with_data data = make false false false data
  let typed_with_data data = make true false false data
  let verbose_with_data data = make true true true data

end

let pr fmt = Printf.printf fmt
let prr s = Printf.printf "%s" s

let rec pr_sep sep f l ctx =
  match l with
  | [x]  -> f x ctx
  | []   -> ()
  | h::t ->
      f h ctx;
      pr sep;
      pr_sep sep f t ctx

let pr_list f l ctx =
  pr_sep ", " f l ctx

let pr_indent (ctx:'a Ctx.t) =
  Printf.printf "\n";
  let rec pr_indent i = match i with
    | 0 -> ()
    | i ->
      Printf.printf "    ";
      pr_indent (i - 1)
  in
  pr_indent ctx.indent

let pr_delim l r f ctx =
  pr l;
  f ctx;
  pr r

let pr_paren f ctx =
  pr_delim "(" ")" f ctx

let pr_brack f ctx =
  pr_delim "[" "]" f ctx

let pr_brace f ctx =
  pr_delim "{" "}" f ctx

let pr_angle f ctx =
  pr_delim "<" ">" f ctx

let pr_quote f ctx =
  pr_delim "\"" "\"" f ctx

and pr_name x _ctx =
  pr "%s" x

let pr_field_opt f (x, a) ctx =
  pr_name x ctx;
  match a with
  | Some a -> 
    pr ": ";
    f a ctx;
  | None -> ()

let pr_field f (x, a) ctx =
  pr_name x ctx;
  pr ": ";
  f a ctx

let rec pr_path xs ctx =
  match xs with
  | [] ->
      ()
  | [h] ->
      pr "%s" h;
  | h::t ->
      pr "%s::" h;
      pr_path t ctx;

and pr_fields_opt f fs ctx = pr_list (pr_field_opt f) fs ctx

and pr_fields f fs ctx = pr_list (pr_field f) fs ctx

and pr_var x ctx =
  pr_name x ctx

and path_to_str xs =
  match xs with
  | [] -> ""
  | x::xs -> Printf.sprintf "::%s%s" x (path_to_str xs);

and pr_decorator d ctx =
  pr "@";
  pr_brace (pr_fields_opt pr_lit d) ctx;
  ctx |> pr_indent;

and pr_annot (x, l) ctx =
  pr_name x ctx;
  pr ":";
  pr_lit l ctx;

and pr_lit l _ctx =
  match l with
  | Ast.LInt (c, Some (true, size)) -> pr "%di%d" c size
  | Ast.LInt (c, Some (false, size)) -> pr "%du%d" c size
  | Ast.LInt (c, None) -> pr "%d" c
  | Ast.LFloat (c, Some size) -> pr "%ff%d" c size;
  | Ast.LFloat (c, None) -> pr "%f" c;
  | Ast.LBool c -> pr "%b" c;
  | Ast.LUnit -> pr "unit";
  | Ast.LString c -> pr "\"%s\"" c
  | Ast.LChar c -> pr "%c" c

