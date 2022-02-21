open Table
open Ast
open Utils

module Ctx = struct
  type t = {
    table: Table.table
  }
  let make = {
    table = PathMap.empty
  }
end

(* [declare] takes an ast and returns a list of top-level declarations *)
let rec declare ast =
  let ctx = Ctx.make in
  let (ctx:Ctx.t) = declare_items ast [] ctx in
  ctx.table

(* TODO: Check for duplicate items *)
and add_decl (xs, d0) (ctx:Ctx.t) : Ctx.t =
  let table = ctx.table |> PathMap.update (xs |> List.rev) (function
  | Some _ -> Some d0
  | None -> Some d0)
  in { table }

and declare_items is xs ctx =
  is |> List.fold_left (fun ctx i -> ctx |> declare_item i xs) ctx

and declare_item i xs ctx =
  match i with
  | IVal (_, x, _, _) ->
      add_decl (x::xs, DItem DGlobal) ctx
  | IEnum (_, x, gs, vs) ->
      let xs = x::xs in
      let arity = (gs |> List.length) in
      let ctx = ctx |> add_decl (xs, DItem (DEnum arity)) in
      vs |> List.fold_left (fun ctx (x, _) -> ctx |> add_decl (x::xs, DItem (DVariant arity))) ctx
  | IExternDef (_, d, gs, _, _) ->
      let x = Ast.def_name d in
      add_decl (x::xs, DItem (DExternDef (List.length gs))) ctx
  | IExternType (_, x, gs) ->
      add_decl (x::xs, DItem (DExternType (List.length gs))) ctx
  | IDef (_, d, gs, _, _, _) ->
      let x = Ast.def_name d in
      add_decl (x::xs, DItem (DDef (List.length gs))) ctx
  | IClass (_, x, gs, ds) ->
      let ctx = ctx |> add_decl (x::xs, DItem (DClass (List.length gs))) in
      ds |> foldl (fun ctx (x, gs, _, _) -> add_decl (x::xs, DItem (DDef (List.length gs))) ctx) ctx
  | IInstance _ -> ctx
  | ITask (_, d, gs, _, _, _) ->
      let x = Ast.def_name d in
      let xs = x::xs in
      let arity = List.length gs in
      add_decl (xs, DItem (DTask arity)) ctx
  | ITypeAlias (_, x, gs, t) ->
      add_decl (x::xs, DItem (DTypeAlias (List.length gs, gs, t))) ctx
  | IMod (_, x, is) ->
      let ctx = add_decl (x::xs, DItem DMod) ctx in
      let xs = x::xs in
      is |> List.fold_left (fun ctx i -> declare_item i xs ctx) ctx
  | IUse (_, xs, alias) -> match alias with
    | Some x -> add_decl (x::xs, DUse xs) ctx
    | None -> add_decl ((List.hd xs)::xs, DUse xs) ctx
