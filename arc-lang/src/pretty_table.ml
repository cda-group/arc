
let rec pr fmt = Printf.printf fmt

and pr_table t =
  pr "Table {\n";
  t |> List.iter (fun (xs, d) ->
    pr "  ";
    Pretty.pr_path xs Pretty.Ctx.brief;
    pr " -> ";
    pr_decl d;
    pr ";\n"
  );
  pr "}\n";

and pr_decl d =
  match d with
  | Table.DItem d -> begin match d with
    | Table.DEnum -> pr "Enum"
    | Table.DExternDef -> pr "Extern Func"
    | Table.DExternType -> pr "Extern Type"
    | Table.DDef -> pr "Func"
    | Table.DTask -> pr "Task"
    | Table.DTypeAlias -> pr "Type Alias"
    | Table.DVariant -> pr "Variant"
    | Table.DGlobal -> pr "Global"
    | Table.DClass -> pr "Class"
    | Table.DMod -> pr "Module"
  end
  | Table.DUse xs -> Pretty.pr_path xs Pretty.Ctx.brief;
