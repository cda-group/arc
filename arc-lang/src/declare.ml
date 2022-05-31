open Utils

module Ctx = struct

  type t = {
    graph: Graph.t
  }

  let make = {
    graph = Graph.make
  }

  and add_node loc (xs, d0) ctx =
    {graph=ctx.graph |> Graph.add loc xs (Graph.DNode d0)}

  and add_edge loc (xs_alias, xs_aliased) ctx =
    {graph=ctx.graph |> Graph.add loc xs_alias (Graph.DEdge xs_aliased)}

end

(* [declare] takes an ast and returns a list of top-level declarations *)
let rec declare ast =
  let ctx = Ctx.make in
  let xs = [] in
  let (ctx:Ctx.t) = ast |> List.fold_left (fun ctx i -> ctx |> declare_item i xs) ctx in
  ctx.graph

(* Declare an item *)
and declare_item i xs ctx =
  match i with
  | Ast.IVal (loc, _, x, _, _) ->
      Ctx.add_node loc (x::xs, Graph.NItem i) ctx
  | Ast.IExternDef (loc, _, _, d, _, _, _, _) ->
      let x = Ast.def_name d in
      Ctx.add_node loc (rev (x::xs), Graph.NItem i) ctx
  | Ast.IExternType (loc, _, x, _, _) ->
      Ctx.add_node loc (rev (x::xs), Graph.NItem i) ctx
  | Ast.IDef (loc, _, _, d, _, _, _, _, _) ->
      let x = Ast.def_name d in
      Ctx.add_node loc (rev (x::xs), Graph.NItem i) ctx
  | Ast.IClass (loc, _, x, _, _, decls) ->
      let ctx = ctx |> Ctx.add_node loc (rev (x::xs), Graph.NItem i) in
      decls |> foldl (fun ctx (x, _, _, _, _) -> Ctx.add_node loc (rev (x::xs), Graph.NItem i) ctx) ctx
  | Ast.IInstance _ -> ctx
  | Ast.ITask (loc, _, d, _, _, _, _, _) ->
      let x = Ast.def_name d in
      Ctx.add_node loc (rev (x::xs), Graph.NItem i) ctx
  | Ast.IType (loc, _, x, _, _, _) ->
      Ctx.add_node loc (rev (x::xs), Graph.NItem i) ctx
  | Ast.IMod (_, _, x, is) ->
      is |> List.fold_left (fun ctx i -> declare_item i (x::xs) ctx) ctx
  | Ast.IUse (loc, _, xs_aliased, alias) ->
      let xs_aliased = match xs_aliased with
      | Ast.PRel xs_aliased -> (rev xs) @ xs_aliased
      | Ast.PAbs xs_aliased -> xs_aliased
      in
      let xs_alias = match alias with
      | Some Ast.UAlias x -> rev (x::xs)
      | Some Ast.UGlob -> todo ()
      | None -> rev ((last xs_aliased)::xs)
      in
      Ctx.add_edge loc (xs_alias, xs_aliased) ctx
