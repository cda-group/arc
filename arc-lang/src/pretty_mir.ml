open Utils
open Pretty
open Pretty_hir

let pr_mir (mir:Mir.mir) =
  let hir = mir |> map (fun ((xs, _), i) -> (xs, i)) in
  let ctx = Ctx.brief in
  hir |> List.iter (fun i -> pr_item i ctx);
  pr "\n";

