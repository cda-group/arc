open Lexer
open Lexing
open Printf
open Utils

let menu () =
  print_endline "Usage:";
  print_endline "$ arc-lang <file> [--debug [verbose]]";
  print_endline "$ cat <file> | arc-lang [--debug [verbose]]"

let print_position outx lexbuf =
  let pos = lexbuf.lex_curr_p in
  fprintf outx "%s:%d:%d" pos.pos_fname pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1)

let print_debug debug stage f data =
    if debug <> Debug.Silent then begin
      printf "[:::%s:::]\n" stage;
      f data
    end

let lexer inx filename =
  let lexbuf = Lexing.from_channel inx in
  lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = filename };
  lexbuf

let get_env key =
  match Sys.getenv_opt key with
  | Some value -> value
  | None -> panic (Printf.sprintf "Environment variable $%s was not set" key)


let compile inx filename debug =

  let lexbuf = lexer inx filename in

  try
    
    let ast = Parser.program Lexer.main lexbuf in
    print_debug debug "AST" (Pretty_ast.pr_ast ast) debug;

    let stdlibpath = get_env "ARC_LANG_STDLIB_PATH" in
    let inx = Core.In_channel.create stdlibpath in
    let lexbuf = lexer inx stdlibpath in

    let stlib = Parser.program Lexer.main lexbuf in

    print_debug debug "AST (Stdlib)" (Pretty_ast.pr_ast stlib) debug;

    let ast = ast @ stlib in

    let table = Declare.declare ast in

    let hir = Ast_to_hir.hir_of_ast table ast in
    print_debug debug "HIR" (Pretty_hir.pr_hir hir) debug;

    let hir = Infer.infer_hir hir debug in
    print_debug debug "HIR (Inferred)" (Pretty_hir.pr_hir hir) debug;

    let mir = Hir_to_mir.mir_of_hir hir in
    print_debug debug "MIR" Pretty_mir.pr_mir mir;

    print_debug debug "Rust" Mir_to_rust.pr_mir mir;

    let mlir = Mir_to_mlir.mlir_of_mir mir in
    print_debug debug "MLIR" Pretty_mlir.pr_mlir (mlir, "");

    let stdlibpath = get_env "ARC_MLIR_STDLIB_PATH" in
    let ch = open_in stdlibpath in
    let stdlib = really_input_string ch (in_channel_length ch) in
    close_in ch;
    Pretty_mlir.pr_mlir (mlir, stdlib);

  with
    | Utils.Compiler_error msg ->
      eprintf "Compiler Error: %s. %s" msg (Printexc.get_backtrace ());
      exit (-1)
    | SyntaxError msg ->
      eprintf "Syntax Error: %a: %s\n" print_position lexbuf msg;
      exit (-1)
    | Parser.Error -> 
      eprintf "Parser Error: %a\n" print_position lexbuf;
      exit (-1)

let main =
    let argv = Core.Sys.get_argv () in
    match argv with
    | [| _ |] ->
        compile Core.In_channel.stdin "stdin" Debug.Silent;
    | [| _; "--debug" |] ->
        compile Core.In_channel.stdin "stdin" Debug.Brief;
    | [| _; "--debug"; "verbose" |] ->
        compile Core.In_channel.stdin "stdin" Debug.Verbose;
    | [| _; filename |] ->
        let inx = Core.In_channel.create filename in
        compile inx filename Debug.Silent;
        Core.In_channel.close inx;
    | [| _; filename; "--debug"; "verbose" |] ->
        let inx = Core.In_channel.create filename in
        compile inx filename Debug.Verbose;
        Core.In_channel.close inx;
    | [| _; filename; "--debug" |] ->
        let inx = Core.In_channel.create filename in
        compile inx filename Debug.Brief;
        Core.In_channel.close inx;
    | _ ->
        menu ()
