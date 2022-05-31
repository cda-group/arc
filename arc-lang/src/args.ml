open Utils

let usage = "
$ arc-lang <file> [OPTIONS]
$ cat <file> | arc-lang [OPTIONS]
"

let get_env key =
  match Sys.getenv_opt key with
  | Some value -> value
  | None -> panic (Printf.sprintf "Environment variable $%s was not set" key)

let input = ref []
let verbose = ref false
let show_parsed = ref false
let show_declared = ref false
let show_desugared = ref false
let show_inferred = ref false
let show_patcomped = ref false
let show_monomorphised = ref false
let show_std = ref false
let show_types = ref false
let show_types_stmts = ref false
let show_externs = ref false
let show_generated = ref false
let show_explicit_paths = ref false
let show_mlir = ref true
let show_mlir_std = ref true

let arc_lang_stdlibpath = get_env "ARC_LANG_STDLIB_PATH"
let arc_mlir_stdlibpath = get_env "ARC_MLIR_STDLIB_PATH"

let debug x = Arg.Tuple ([Arg.Clear show_mlir; Arg.Clear show_mlir_std; Arg.Set x])

let speclist = [
  ("--verbose", debug verbose, "Print extra info");
  ("--show-parsed", debug show_parsed, "Show AST");
  ("--show-declared", debug show_declared, "Show declaration graph");
  ("--show-desugared", debug show_desugared, "Show desugared IR");
  ("--show-inferred", debug show_inferred, "Show inferred IR");
  ("--show-patcomped", debug show_patcomped, "Show pattern compiled IR");
  ("--show-monomorphised", debug show_monomorphised, "Show monomorhpised IR");
  ("--show-std", debug show_std, "Show Std");
  ("--show-types", debug show_types, "Show types");
  ("--show-types-stmts", debug show_types_stmts, "Show types of statements");
  ("--show-externs", debug show_externs, "Show extern defs");
  ("--show-explicit-paths", debug show_explicit_paths, "Show explicit paths");
]

let anon_fun filepath = input := filepath::!input

let parse () =
  Arg.parse speclist anon_fun usage;
