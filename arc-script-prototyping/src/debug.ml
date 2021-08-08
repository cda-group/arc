let debug_constraints cs hir =
  Printf.printf "\nConstraints:\n";
  cs |> List.iter (fun (t0, t1) ->
    Pretty_hir.pr_type t0 (Pretty_hir.Ctx.make hir true);
    Printf.printf " <=> ";
    Pretty_hir.pr_type t1 (Pretty_hir.Ctx.make hir true);
    Printf.printf ",\n";
  );

and debug_substitutions ss hir =
  Printf.printf "\nSubstitutions:\n";
  ss |> List.iter (fun (x, t) ->
    Printf.printf "'%s" x;
    Printf.printf " => ";
    Pretty_hir.pr_type t (Pretty_hir.Ctx.make hir true);
    Printf.printf ",\n";
  );
(*   Printf.printf "----------\n"; *)
