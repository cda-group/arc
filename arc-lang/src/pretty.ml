
module Ctx = struct
  type 'a t = {
    indent: int;
    show_types: bool;
    show_externs: bool;
    show_parens: bool;
  }

  let indent ctx =
    { ctx with indent = ctx.indent + 1 }

  let make show_types show_externs show_parens = {
      indent = 0;
      show_types;
      show_externs;
      show_parens;
    }

  let brief = make false false false
  let typed = make true false false
  let verbose = make true true true

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

let rec pr_path xs ctx =
  match xs with
  | [] ->
      ()
  | [h] ->
      pr "%s" h;
  | h::t ->
      pr "%s::" h;
      pr_path t ctx;

and pr_field_opt f (x, a) ctx =
  pr_name x ctx;
  match a with
  | Some a -> 
    pr ": ";
    f a ctx;
  | None -> ()

and pr_field f (x, a) ctx =
  pr_name x ctx;
  pr ": ";
  f a ctx;

and pr_var x ctx =
  pr_name x ctx

