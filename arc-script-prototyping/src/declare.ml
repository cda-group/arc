open Table
open Ast

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
  | IVal (x, _, _) ->
      add_decl (x::xs, DItem DGlobal) ctx
  | IEnum (x, _, vs) ->
      let xs = x::xs in
      let ctx = add_decl (xs, DItem DEnum) ctx in
      vs |> List.fold_left (fun ctx (x, _) -> add_decl (x::xs, DItem DVariant) ctx) ctx
  | IExternFunc (x, _, _, _) ->
      add_decl (x::xs, DItem DExternFunc) ctx
  | IExternType (x, _) ->
      add_decl (x::xs, DItem DExternType) ctx
  | IFunc (x, _, _, _, _) ->
      add_decl (x::xs, DItem DFunc) ctx
  | ITask (x, _, _, i0, i1, _) ->
      let xs = x::xs in
      add_decl (xs, DItem DTask) ctx
        |> declare_interface i0 ("I"::xs)
        |> declare_interface i1 ("O"::xs)
  | ITypeAlias (x, _) ->
      add_decl (x::xs, DItem DTypeAlias) ctx
  | IMod (x, is) ->
      let ctx = add_decl (x::xs, DItem DMod) ctx in
      let xs = x::xs in
      is |> List.fold_left (fun ctx i -> declare_item i xs ctx) ctx
  | IUse (xs, alias) -> match alias with
    | Some x -> add_decl (x::xs, DUse xs) ctx
    | None -> add_decl ((List.hd xs)::xs, DUse xs) ctx

and declare_interface i xs ctx =
  let ctx = add_decl (xs, DItem DEnum) ctx in
  match i with
  | PSingle _ ->
      add_decl ("_"::xs, DItem DVariant) ctx
  | PTagged ps ->
      ps |> List.fold_left (fun ctx (x, _) -> add_decl (x::xs, DItem DVariant) ctx) ctx
