open Utils

module Ctx = struct
  type t = {
    indent: int;
    debug: bool;
  }

  let print_indent ctx: unit =
    Printf.printf "\n";
    let rec print_indent = function
      | 0 -> ()
      | i ->
        Printf.printf "    ";
        print_indent (i - 1)
    in
    print_indent ctx.indent

  let indent ctx =
    { ctx with indent = ctx.indent + 1 }

  and make debug = { indent = 0; debug; }

end

let pr fmt = Printf.printf fmt
let prr s = Printf.printf "%s" s

let quoted f s =
  pr "\"";
  f s;
  pr "\""

let paren f s =
  pr "(";
  f s;
  pr ")"

let rec pr_sep sep f l (ctx:Ctx.t) =
  match l with
  | [x]  -> f x ctx
  | []   -> ()
  | h::t ->
      f h ctx;
      pr sep;
      pr_sep sep f t ctx

let rec pr_mlir (mlir:Mlir.mlir) =
  let ctx = Ctx.make false in
  pr "\nmodule @toplevel {";
  let ctx' = ctx |> Ctx.indent in
  mlir |> List.iter (fun i -> pr_item i ctx');
  ctx |> Ctx.print_indent;
  pr "}\n";

and pr_item (xs, i) ctx =
  ctx |> Ctx.print_indent;
  match i with
  | Mlir.IAssign _ ->
      todo ()
  | Mlir.IExternFunc (ps, t) ->
      pr "func private @";
      pr_path xs;
      pr_params ps ctx;
      pr " -> ";
      pr_type t ctx;
  | Mlir.IFunc (ps, t, b) ->
      pr "func @";
      pr_path xs;
      pr_params ps ctx;
      pr " -> ";
      pr_type t ctx;
      pr " ";
      pr_block b ctx;
  | Mlir.ITask (ps, b) ->
      pr "func @";
      pr_path xs;
      pr_params ps ctx;
      pr " ";
      pr_block b ctx;

and pr_params ps ctx =
  pr "(";
  pr_sep ", " pr_param ps ctx;
  pr ")";

and pr_param (x, t) ctx =
  prr "%";
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_variant_type (x, t) (ctx:Ctx.t) =
  pr_path x;
  pr ": ";
  pr_type t ctx;

and pr_path x =
  pr "%s" x;

and pr_type t ctx =
  match t with
  | Mlir.TFunc (ts, t) ->
      pr "(";
      pr_sep ", " pr_type ts ctx; 
      pr ") -> ";
    pr_type t ctx;
  | Mlir.TRecord fts ->
      pr "!arc.struct<";
      pr_sep ", " pr_field_type fts ctx;
      pr ">";
  | Mlir.TEnum vts ->
      pr "!arc.enum<";
      pr_sep ", " pr_variant_type vts ctx;
      pr ">";
  | Mlir.TAdt (xs, _ts) ->
      pr "!arc.adt<";
      pr_path xs;
      pr ">";
  | Mlir.TStream t ->
      pr "!arc.stream<";
      pr_type t ctx;
      pr ">";
  | Mlir.TNative x ->
      pr "%s" x;

and pr_block ss ctx =
  pr "{";
  pr_sep "" pr_ssa ss (ctx |> Ctx.indent);
  ctx |> Ctx.print_indent;
  pr "}";

and pr_lhs lhs ctx =
  match lhs with
  | Some (v, _) ->
    pr_var v ctx;
    pr " = ";
  | _ -> ()

and pr_ssa (lhs, e) ctx =
  ctx |> Ctx.print_indent;
  pr_lhs lhs ctx;

  match e with
  | Mlir.EAccess (a0, x1) ->
      quoted pr "arc.struct_access";
      paren (pr_arg_var a0) ctx;
      pr "{ field = ";
      quoted (pr_name x1) ctx;
      pr " } :";
      paren (pr_arg_type a0) ctx;
      pr "-> ";
      pr_lhs_type lhs ctx;
  | Mlir.EBinOp (op, a0, a1) ->
      begin match op with
      | Mlir.BAdd Mlir.NInt   -> pr "arc.addi"
      | Mlir.BAdd Mlir.NFlt   -> pr "addf"
      | Mlir.BSub Mlir.NInt   -> pr "arc.subi"
      | Mlir.BSub Mlir.NFlt   -> pr "subf"
      | Mlir.BMul Mlir.NInt   -> pr "arc.muli"
      | Mlir.BMul Mlir.NFlt   -> pr "mulf"
      | Mlir.BDiv Mlir.NInt   -> pr "arc.divi"
      | Mlir.BDiv Mlir.NFlt   -> pr "divf"
      | Mlir.BMod Mlir.NInt   -> pr "arc.remi"
      | Mlir.BMod Mlir.NFlt   -> pr "remf"
      | Mlir.BPow Mlir.NInt   -> pr "arc.powi"
      | Mlir.BPow Mlir.NFlt   -> pr "math.powf"
      | Mlir.BLt  Mlir.NInt   -> pr "arc.cmpi lt,"
      | Mlir.BLt  Mlir.NFlt   -> pr "cmpf olt,"
      | Mlir.BLeq Mlir.NInt   -> pr "arc.cmpi le,"
      | Mlir.BLeq Mlir.NFlt   -> pr "cmpf ole,"
      | Mlir.BGt  Mlir.NInt   -> pr "arc.cmpi gt,"
      | Mlir.BGt  Mlir.NFlt   -> pr "cmpf ogt,"
      | Mlir.BGeq Mlir.NInt   -> pr "arc.cmpi ge,"
      | Mlir.BGeq Mlir.NFlt   -> pr "cmpf oge,"
      | Mlir.BEqu Mlir.EqInt  -> pr "arc.cmpi eq,"
      | Mlir.BEqu Mlir.EqFlt  -> pr "cmpf oeq,"
      | Mlir.BEqu Mlir.EqBool -> pr "cmpi eq,"
      | Mlir.BNeq Mlir.EqInt  -> pr "arc.cmpi ne,"
      | Mlir.BNeq Mlir.EqFlt  -> pr "cmpf one,"
      | Mlir.BNeq Mlir.EqBool -> pr "cmpi ne,"
      | Mlir.BAnd             -> pr "and"
      | Mlir.BOr              -> pr "or"
      | Mlir.BXor             -> pr "xor"
      | Mlir.BBand            -> pr "arc.and"
      | Mlir.BBor             -> pr "arc.or"
      | Mlir.BBxor            -> pr "arc.xor"
      | _ -> panic "Undefined op"
      end;
      pr " ";
      pr_arg_var a0 ctx;
      pr ", ";
      pr_arg_var a1 ctx;
      pr " : ";
      pr_arg_type a0 ctx;
  | Mlir.ECall (a0, args) ->
      pr_arg_var a0 ctx;
      paren (pr_sep ", " pr_arg_var args) ctx;
      pr " : ";
      pr_arg_type a0 ctx;
      pr " -> ";
      pr_lhs_type lhs ctx;
  | Mlir.EReceive a0 ->
      pr "arc.receive";
      paren (pr_arg_var a0) ctx;
      pr " : ";
      pr_arg_type a0 ctx;
      pr " -> ";
      paren (pr_lhs_type lhs) ctx;
  | Mlir.EEmit (a0, a1) ->
      pr "arc.emit";
      paren (fun ctx ->
        pr_arg_var a0 ctx;
        prr ", ";
        pr_arg_var a1 ctx;
      ) ctx;
      pr " : ";
      pr_arg_type a1 ctx;
      pr " -> ";
      pr_lhs_type lhs ctx;
  | Mlir.EEnwrap (xs, a0) ->
      pr "arc.make_enum";
      begin match a0 with
      | None -> pr "()"
      | Some a0 -> paren (pr_arg a0) ctx;
      end;
      pr " as ";
      quoted pr_path xs;
      pr " : ";
      pr_lhs_type lhs ctx;
  | Mlir.EIs (xs, a0) ->
      pr "arc.enum_check";
      paren (pr_arg a0) ctx;
      pr " is ";
      quoted pr_path xs;
      pr " : ";
      pr_lhs_type lhs ctx;
  | Mlir.EUnwrap (xs, a0) ->
      pr "arc.enum_access ";
      quoted pr_path xs;
      pr " in ";
      paren (pr_arg a0) ctx;
      pr " : ";
      begin match lhs with
      | Some (_, t) -> pr_type t ctx
      | None -> pr "none"
      end;
  | Mlir.EIf (a0, b0, b1) ->
      quoted pr "arc.if";
      paren (pr_arg_var a0) ctx;
      paren (fun ctx ->
        pr_block b0 ctx;
        pr ",";
        pr_block b1 ctx;
      ) ctx;
      pr " : (i1) -> ";
      pr_lhs_type lhs ctx;
  | Mlir.EConst c ->
      begin match c with
      | Mlir.CInt d -> pr "arc.constant %d : i32" d
      | Mlir.CFloat f -> pr "constant %f : f32" f
      | Mlir.CBool b -> pr "constant %b : i0" b
      | Mlir.CFun x -> pr "constant @%s" x
      end;
  | Mlir.ELoop b ->
      pr "scf.while : () -> () {";
      ctx |> Ctx.indent |> Ctx.print_indent;
      pr "%%condition = constant 1 : i1";
      ctx |> Ctx.indent |> Ctx.print_indent;
      pr "scf.condition(%%condition)";
      ctx |> Ctx.print_indent;
      pr "} do ";
      pr_block b ctx;
  | Mlir.ERecord fas ->
      quoted pr "arc.make_struct";
      paren (pr_sep ", " pr_field_expr fas) ctx;
      pr " : ";
      pr_lhs_type lhs ctx;
  | Mlir.EReturn a0 ->
      begin match a0 with
      | Some a0 ->
          pr "return ";
          pr_arg_var a0 ctx;
          pr " : ";
          pr_arg_type a0 ctx;
      | None ->
          pr "return"
      end
  | Mlir.EResult a0 ->
      quoted pr "arc.block.result";
      begin match a0 with
      | Some a0 ->
          paren (pr_arg_var a0) ctx;
          pr " : ";
          paren (pr_arg_type a0) ctx;
          pr " -> ";
          pr_lhs_type lhs ctx;
      | None -> ()
      end
  | Mlir.EBreak a0 ->
      quoted pr "arc.loop.break";
      begin match a0 with
      | Some a0 ->
          paren (pr_arg_var a0) ctx;
          pr " : ";
          paren (pr_arg_type a0) ctx;
          pr " -> ";
          pr_lhs_type lhs ctx;
      | None -> pr ": () -> ()";
      end
  | Mlir.EContinue ->
      quoted pr "arc.loop.continue";
      pr "() : () -> ()"
  | Mlir.EYield ->
      pr "scf.yield";
      pr " : () -> ()"
  | Mlir.ENoop ->
      pr "// noop"

and pr_name x _ctx =
  pr "%s" x;

and pr_arg a ctx =
  pr_arg_var a ctx;
  pr " : ";
  pr_arg_type a ctx

and pr_arg_var (v, _) ctx =
  pr_var v ctx;

and pr_arg_type (_, t) ctx =
  pr_type t ctx;

and pr_var v _ctx =
  prr "%";
  pr "%s" v;

and pr_lhs_type v ctx =
  match v with
  | Some (_, t) -> pr_type t ctx
  | None -> pr "()"

and pr_field_type (x, t) ctx =
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_field_expr (x, v) ctx =
  pr_name x ctx;
  pr ": ";
  pr_arg_var v ctx;
