open Utils

let add_item i hir = i::hir

let rec nominal x ts = Ast.TPath ([x], ts)

and generic x = Ast.TPath ([x], [])

and add_cell_intrinsics ast =
  let t_cell = (nominal "Cell" [generic "T"]) in
  let t_unit = (nominal "unit" []) in
  let t_elem = generic "T" in
  let p_cell = ("cell", t_cell) in
  let p_elem = ("elem", t_elem) in
  ast |> add_item (Ast.IExternType ("Cell", ["T"]))
      |> add_item (Ast.IExternFunc ("new_cell", ["T"], [p_elem], t_cell))
      |> add_item (Ast.IExternFunc ("update_cell", ["T"], [p_cell; p_elem], t_unit))
      |> add_item (Ast.IExternFunc ("read_cell", ["T"], [p_cell], t_cell))

and add_scalar ast x =
  ast |> add_item (Ast.IExternType (x, []))

and add_scalar_intrinsics ast =
  ["unit"; "i16"; "i32"; "i64"; "i128"; "f32"; "f64"; "bool"; "char"; "string"] |> foldl add_scalar ast

and add_intrinsics ast =
  ast |> add_scalar_intrinsics
      |> add_cell_intrinsics
