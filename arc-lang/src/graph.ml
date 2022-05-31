module PathMap = Map.Make(
  struct
    type t = Ir1.path
    let compare = Stdlib.compare
  end
)

type decl =
  | DNode of node
  | DEdge of Ir1.path (* An alias *)
and node =
  | NItem of Ast.item
  | NMethodDecl of Ast.method_decl
and name = string
and t = decl PathMap.t

let rec add loc xs d graph =
  graph |> PathMap.update xs (function
  | Some _ -> raise (Error.NamingError (loc, Printf.sprintf "Duplicate declaration: %s" (Print.path_to_str xs)))
  | None -> Some d)

and resolve_path xs graph =
  if !Args.verbose then begin
    Printf.printf "Resolving %s\n" (Print.path_to_str xs)
  end;
  match graph |> PathMap.find_opt xs with
  | Some DEdge xs -> graph |> resolve_path xs
  | Some DNode i -> Some (xs, i)
  | None -> None

and make = PathMap.empty

and debug graph =
  graph |> PathMap.iter debug_entry

and debug_entry xs n =
  Print.pr_path xs Print.Ctx.make;
  match n with
  | DNode n ->
      begin match n with
      | NItem _ ->
          Print.pr " -> Item\n";
          (* Print.Ast.pr_item i Print.Ctx.brief; *)
          (* Print.pr "\n"; *)
      | NMethodDecl _ ->
          Print.pr " -> MethodDecl\n";
      end
  | DEdge xs ->
      Print.pr " -> Path(";
      Print.pr_path xs Print.Ctx.make;
      Print.pr ")\n";
