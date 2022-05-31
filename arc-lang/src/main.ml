open Lexing
open Printf
open Utils

let file_to_mod (file, inx) =
  let lexbuf = Lexing.from_channel inx in
  lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = file };
  try
    let items = Parser.program Lexer.main lexbuf in
    Ast.IMod (NoLoc, [], file, items)
  with
  | Error.ParsingError (loc, msg) -> Error.report loc msg
  | Error.LexingError (loc, msg) -> Error.report loc msg
  | Parser.Error -> Error.report (Lexer.loc lexbuf) "Parser Error"

let compile ast =
  try
    if !Args.show_parsed then Print.Ast.pr_ast ast;

    let graph = Declare.declare ast in
    if !Args.show_declared then Graph.debug graph;

    let ir1 = Ast_to_ir1.ast_to_ir1 graph ast in
    if !Args.show_desugared then Print.Ir1.pr_ir1 ir1;

    let ir1 = Infer_ir1.infer_ir1 ir1 in
    if !Args.show_inferred then Print.Ir1.pr_ir1 ir1;

    let ir2 = Ir1_to_ir2.ir1_to_ir2 ir1 in
    if !Args.show_patcomped then Print.Ir2.pr_ir2 ir2;

    let ir3 = Ir2_to_ir3.ir2_to_ir3 ir2 in
    if !Args.show_monomorphised then Print.Ir3.pr_ir3 ir3;

    let mlir = Ir3_to_mlir.ir3_to_mlir ir3 in
    if !Args.show_mlir then Print.Mlir.pr_mlir mlir;
    ()
  with
  | Error.TypingError (loc, msg) -> Error.report loc msg
  | Error.NamingError (loc, msg) -> Error.report loc msg

let check_duplicates input =
    let d = Utils.duplicates String.compare input in
    if d <> [] then
      let d = d |> String.concat ", " in
      raise (Error.InputError (sprintf "Found duplicate file-modules: %s" d))
    else
      ()

let filepath_to_modname path =
  path
    |> Filename.basename
    |> Filename.remove_extension
    |> Str.global_replace (Str.regexp "-") "_"

let main =
  Args.parse ();
  try
    let input = match !Args.input with
    | [] -> [("stdin", Core.In_channel.stdin)]
    | input -> input |> List.map (function i -> (filepath_to_modname i, Core.In_channel.create i))
    in
    let input = ("std", Core.In_channel.create Args.arc_lang_stdlibpath)::input in
    check_duplicates (input |> map fst);
    let ast = input |> map (fun file -> file_to_mod file) in
    let ast = match ast with
    | [] -> unreachable ()
    | std_mod::user_mods ->
        let std_mod = if !Args.show_std then
          std_mod
        else
          Ast.std_loc std_mod
        in
        let prelude = Ast.extract_prelude std_mod in
        let user_mods = user_mods |> List.map (function i -> Ast.add_prelude prelude i) in
        std_mod::user_mods
    in
    compile ast;
  with
  | Utils.Panic msg ->
      eprintf "Panic: %s. %s" msg (Printexc.get_backtrace ());
      exit 1
  | Error.InputError msg ->
      eprintf "InputError: %s" msg;
      exit 1
