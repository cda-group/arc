open Utils
open Pretty

let rec pr_mlir mlir =
  let ctx = Ctx.make in
  ctx |> pr_indent;
  pr "module @toplevel {";
  if !Args.show_mlir_std then begin
    ctx |> pr_indent;
    let ch = open_in Args.arc_mlir_stdlibpath in
    prr (really_input_string ch (in_channel_length ch));
    close_in ch;
  end;
  let ctx' = ctx |> Ctx.indent in
  pr_items mlir ctx';
  ctx |> pr_indent;
  pr "}\n";

and show_item (_, i) =
  match Mlir.item_loc i with
  | LocStd _ | NoLocStd when not !Args.show_std -> false
  | _ -> true

and pr_items is ctx =
  is |> filter show_item |> List.iter (fun i -> pr_item i ctx);

and pr_item (x, i) ctx =
  ctx |> pr_indent;
  match i with
  | Mlir.IExternFunc (_, x_rust, async, ps, t) ->
      pr "func.func private ";
      pr_symbol x ctx;
      pr_params ps ctx;
      pr " -> ";
      pr_type t ctx;
      if async then 
        pr " attributes { rust.async, rust.annotation=\"#[rewrite(unmangled = \\\"%s\\\")]\"}" x_rust
      else
        pr " attributes { rust.declare, rust.annotation=\"#[rewrite(unmangled = \\\"%s\\\")]\"}" x_rust
  | Mlir.IFunc (_, ps, t, b) ->
      pr "func.func ";
      pr_symbol x ctx;
      pr_params ps ctx;
      pr_opt pr_return_type t ctx;
      pr " attributes { rust.declare";
      begin if x = "main" then
        pr ", rust.annotation = \"#[rewrite(main)]\"";
      end;
      pr " } ";
      pr_block b ctx;

and pr_params ps ctx =
  pr_paren (pr_list pr_param ps) ctx

and pr_param (x, t) ctx =
  prr "%";
  pr_name x ctx;
  pr ": ";
  pr_type t ctx;

and pr_symbol x ctx =
  pr "@";
  pr_name x ctx

and pr_path x _ctx =
  pr "%s" x;

and pr_types ts ctx =
  pr_list pr_type ts ctx

and pr_return_type t ctx =
  pr " -> ";
  begin match t with
  | Mlir.TFunc _ -> pr_paren (pr_type t) ctx
  | _ -> pr_type t ctx;
  end

and pr_type t ctx =
  match t with
  | Mlir.TFunc (ts, t) ->
      pr_paren (pr_types ts) ctx;
      pr_return_type t ctx;
  | Mlir.TRecord fts ->
      pr "!arc.struct";
      pr_angle (pr_list (pr_field pr_type) fts) ctx;
  | Mlir.TEnum vts ->
      pr "!arc.enum";
      pr_angle (pr_list (pr_field pr_type) vts) ctx;
  | Mlir.TAdt (x, []) ->
      pr "!arc.adt";
      pr "<";
      pr_quote (pr_path x) ctx;
      pr ">";
  | Mlir.TAdt (x, ts) ->
      pr "!arc.generic_adt";
      pr "<";
      pr_quote (pr_path x) ctx;
      pr ", ";
      pr_types ts ctx;
      pr ">";
  | Mlir.TNative x ->
      pr "%s" x;

and pr_block ss ctx =
  pr "{";
  pr_sep "" pr_ssa ss (ctx |> Ctx.indent);
  ctx |> pr_indent;
  pr "}";

and pr_lhs lhs ctx =
  match lhs with
  | Some (v, _) ->
    pr_var v ctx;
    pr " = ";
  | _ -> ()

and pr_ssa (lhs, e) ctx =
  ctx |> pr_indent;
  match e with
  | _ ->
    pr_lhs lhs ctx;
    match e with
    | Mlir.OAccess (_, a0, x1) ->
        pr_quote pr "arc.struct_access";
        pr_paren (pr_arg_var a0) ctx;
        pr " { field = ";
        pr_quote (pr_name x1) ctx;
        pr " } : ";
        pr_paren (pr_arg_type a0) ctx;
        pr " -> ";
        pr_lhs_type lhs ctx;
    | Mlir.OUpdate (_, _a0, _x1, _a1) ->
        todo ()
    | Mlir.OCallExpr (_, a0, args) ->
        pr "call_indirect ";
        pr_arg_var a0 ctx;
        pr_paren (pr_list pr_arg_var args) ctx;
        pr " : ";
        pr_arg_type a0 ctx;
    | Mlir.OCallItem (_, x, args) ->
        pr "call ";
        pr_symbol x ctx;
        pr_paren (pr_list pr_arg_var args) ctx;
        pr " : ";
        pr_lhs_type lhs ctx;
        pr " -> ";
        pr_paren (pr_list pr_arg_type args) ctx;
    | Mlir.OEnwrap (_, x, a0) ->
        pr "arc.make_enum";
        begin match a0 with
        | None -> pr "()"
        | Some a0 -> pr_paren (pr_arg a0) ctx;
        end;
        pr " as ";
        pr_quote (pr_path x) ctx;
        pr " : ";
        pr_lhs_type lhs ctx;
    | Mlir.OCheck (_, x, a0) ->
        pr "arc.enum_check";
        pr_paren (pr_arg a0) ctx;
        pr " is ";
        pr_quote (pr_path x) ctx;
        pr " : ";
        pr_lhs_type lhs ctx;
    | Mlir.OUnwrap (_, x, a0) ->
        pr "arc.enum_access ";
        pr_quote (pr_path x) ctx;
        pr " in ";
        pr_paren (pr_arg a0) ctx;
        pr " : ";
        begin match lhs with
        | Some (_, t) -> pr_type t ctx
        | None -> pr "none"
        end;
    | Mlir.OIf (_, a0, b0, b1) ->
        pr_quote pr "arc.if";
        pr_paren (pr_arg_var a0) ctx;
        pr_paren (fun ctx ->
          pr_block b0 ctx;
          pr ",";
          pr_block b1 ctx;
        ) ctx;
        pr " : (i1) -> ";
        pr_lhs_type lhs ctx;
    | Mlir.OConst (_, c) ->
        begin match c with
        | Mlir.CInt d ->
            pr "arc.constant %d : " d;
            pr_lhs_type lhs ctx;
        | Mlir.CFloat f ->
            pr "arith.constant %f : " f;
            pr_lhs_type lhs ctx;
        | Mlir.CBool b ->
            pr "arith.constant %b" b;
        | Mlir.CFun x ->
            pr "constant ";
            pr_symbol x ctx;
            pr " : ";
            pr_lhs_type lhs ctx;
        | Mlir.CAdt s ->
            pr "arc.adt_constant \"%s\" : " s;
            pr_lhs_type lhs ctx;
        end;
    | Mlir.OLoop (_, b) ->
        pr "scf.while : () -> () {";
        ctx |> Ctx.indent |> pr_indent;
        pr "%%condition = constant 1 : i1";
        ctx |> Ctx.indent |> pr_indent;
        pr "scf.condition(%%condition)";
        ctx |> pr_indent;
        pr "} do ";
        pr_block b ctx;
    | Mlir.ORecord (_, vs0, ts0) ->
        pr "arc.make_struct";
        pr_paren (pr_vars_types vs0 ts0) ctx;
        pr " : ";
        pr_lhs_type lhs ctx;
    | Mlir.OReturn (_, a0) ->
        begin match a0 with
        | Some a0 ->
            pr "return ";
            pr_arg_var a0 ctx;
            pr " : ";
            pr_arg_type a0 ctx;
        | None ->
            pr "return"
        end
    | Mlir.OResult (_, a0) ->
        pr_quote pr "arc.block.result";
        begin match a0 with
        | Some a0 ->
            pr_paren (pr_arg_var a0) ctx;
            pr " : ";
            pr_paren (pr_arg_type a0) ctx;
            pr " -> ";
            pr_lhs_type lhs ctx;
        | None ->
            pr "() : () -> ()"
        end
    | Mlir.OBreak (_, a0) ->
        pr_quote pr "arc.loop.break";
        begin match a0 with
        | Some a0 ->
            pr_paren (pr_arg_var a0) ctx;
            pr " : ";
            pr_paren (pr_arg_type a0) ctx;
            pr " -> ";
            pr_lhs_type lhs ctx;
        | None -> pr ": () -> ()";
        end
    | Mlir.OContinue _ ->
        pr_quote pr "arc.loop.continue";
        pr "() : () -> ()"
    | Mlir.OYield _ ->
        pr "scf.yield";
        pr " : () -> ()"
    | Mlir.OSpawn (_, xs, args) ->
        pr "arc.spawn(";
        pr "@";
        pr_path xs ctx;
        pr ",";
        pr_list pr_arg args ctx;
        pr ")";

and pr_vars_types vs0 ts0 ctx =
  pr_list pr_var vs0 ctx;
  pr " : ";
  pr_list pr_type ts0 ctx;

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

