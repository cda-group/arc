open Core
open Lexer
open Lexing

let menu () = printf "Usage: arc-script [<file>]\n"

let print_position outx lexbuf =
  let pos = lexbuf.lex_curr_p in
  fprintf outx "%s:%d:%d" pos.pos_fname
    pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1)

let main =
    let argv = Sys.get_argv () in
    if Array.length argv < 2 then
      menu ()
    else
      match argv.(1) with
      | filename ->
          let inx = In_channel.create filename in

          let lexbuf = Lexing.from_channel inx in
          lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = filename };
          begin try
            
            let ast = Parser.program Lexer.main lexbuf in
            print_endline "\n\n[:::AST:::]";
            Pretty_ast.pr_ast ast;

            let ast = Intrinsics.add_intrinsics ast in

            print_endline "\n\n[:::AST + Intrinsics:::]";
            Pretty_ast.pr_ast ast;

            let table = Declare.declare ast in

            let hir = Ast_to_hir.hir_of_ast table ast in

            print_endline "\n\n[:::HIR:::]";
            Pretty_hir.pr_hir hir;

            let hir = Infer.infer_hir hir in

            print_endline "\n\n[:::HIR (Inferred):::]";
            Pretty_hir.pr_hir hir;

            let mir = Hir_to_mir.mir_of_hir hir in

            print_endline "\n\n[:::Monomorphised IR:::]";
            Pretty_mir.pr_mir mir;

            let mlir = Mir_to_mlir.mlir_of_thir mir in

            print_endline "\n\n[:::MLIR:::]";
            Pretty_mlir.pr_mlir mlir;

          with
            | Utils.Compiler_error msg ->
              Printf.eprintf "Compiler Error: %s. %s" msg (Printexc.get_backtrace ())
            | SyntaxError msg ->
              fprintf stderr "Syntax Error: %a: %s\n" print_position lexbuf msg;
              exit (-1)
            | Parser.Error -> 
              fprintf stderr "Parser Error: %a\n" print_position lexbuf;
              exit (-1)
          end;
          In_channel.close inx
