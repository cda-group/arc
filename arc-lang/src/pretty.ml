
module Ctx = struct
  type 'a t = {
    indent: int;
  }

  let indent ctx = {
    indent = ctx.indent + 1
  }

  let make = {
    indent = 0
  }

end

let pr fmt = Printf.printf fmt
let prr s = Printf.printf "%s" s
let prln s = Printf.printf "%s\n" s

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

let pr_fields_opt f fs ctx = pr_list (pr_field_opt f) fs ctx

let pr_field f (x, a) ctx =
  pr_name x ctx;
  pr ": ";
  f a ctx

let pr_variant f (x, a) ctx =
  pr_name x ctx;
  f a ctx

let pr_fields f fs ctx = pr_list (pr_field f) fs ctx

let rec pr_path xs ctx =
  pr_sep "::" pr_name xs ctx;

and path_to_str xs =
  String.concat "::" xs

and pr_var x ctx =
  pr_name x ctx

and pr_decorator d ctx =
  if d <> [] then begin
    pr "@";
    pr_brace (pr_fields_opt pr_lit d) ctx;
    ctx |> pr_indent;
  end

and pr_annot (x, l) ctx =
  pr_name x ctx;
  pr ":";
  pr_lit l ctx;

and pr_lit l _ctx =
  match l with
  | Ast.LInt (_, c, Some s) -> pr "%d%s" c s;
  | Ast.LInt (_, c, None) -> pr "%d" c
  | Ast.LFloat (_, c, Some s) -> pr "%f%s" c s;
  | Ast.LFloat (_, c, None) -> pr "%f" c;
  | Ast.LBool (_, c) -> pr "%b" c;
  | Ast.LUnit _ -> pr "unit";
  | Ast.LString (_, c) -> pr "\"%s\"" c
  | Ast.LChar (_, c) -> pr "%c" c

and pr_nl f ctx =
  f ctx;
  Printf.printf "\n"

and pr_tail f a ctx =
  begin match a with
  | Some a ->
      pr "|";
      f a ctx;
  | None ->
      ()
  end

and pr_opt f a ctx =
  match a with
  | Some a ->
      f a ctx;
  | None ->
      ()

and pr_each fs ctx =
  fs |> List.iter (fun f -> f ctx)

and pr_record f (fields, tail) ctx =
  pr "#";
  pr_brace (pr_each [
    pr_list (pr_field_opt f) fields;
    pr_tail f tail;
  ]) ctx;

and pr_record_explicit f (fields, tail) ctx =
  pr "#";
  pr_brace (pr_each [
    pr_list (pr_field f) fields;
    pr_tail f tail;
  ]) ctx

and pr_enum f (fields, tail) ctx =
  pr "#";
  pr_brace (pr_each [
    pr_list (pr_variant f) fields;
    pr_tail f tail;
  ]) ctx;
