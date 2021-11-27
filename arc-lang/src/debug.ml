let debug_substitutions ss =
  Printf.printf "\nSubstitutions:\n";
  ss |> List.iter (fun (x, t) ->
    Printf.printf "'%s" x;
    Printf.printf " => ";
    Pretty_hir.pr_type t Pretty.Ctx.brief;
    Printf.printf ",\n";
  );
(*   Printf.printf "----------\n"; *)
