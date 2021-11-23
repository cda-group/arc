
let rec pr fmt = Printf.printf fmt

and pr_table t =
  pr "Table {\n";
  t |> List.iter (fun (xs, d) ->
    pr "  ";
    Pretty_hir.pr_path xs;
    pr " -> ";
    pr_decl d;
    pr ";\n"
  );
  pr "}\n";

and pr_decl d =
  match d with
  | Table.DItem d -> begin match d with
    | Table.DEnum -> pr "Enum"
    | Table.DExternFunc -> pr "Extern Func"
    | Table.DExternType -> pr "Extern Type"
    | Table.DFunc -> pr "Func"
    | Table.DTask -> pr "Task"
    | Table.DTypeAlias -> pr "Type Alias"
    | Table.DVariant -> pr "Variant"
    | Table.DGlobal -> pr "Global"
  end
  | Table.DUse xs -> Pretty_hir.pr_path xs;
