open Utils
open Error

type ast = items

and names = name list
and name = string
and defname =
  | DName of name
  | DUnOp of unop * tys
  | DBinOp of binop * tys
and paths = path list
and path =
  | PAbs of names
  | PRel of names
and arms = arm list
and arm = pattern * expr
and params = param list
and param = pattern * ty option
and sinks = sink list
and sink = name * ty option
and index = int
and 't fields = 't field list
and 't field = name * 't option
and 't record = 't fields * 't option
and 't variants = 't variant list
and 't variant = name * 't
and 't enum = 't variants * 't option
and block = stmts * expr option
and generics = generic list
and generic = name

and decorator = lit fields

and async = bool

and items = item list
and item =
  | IExternDef  of loc * decorator * async * defname * generics * tys * ty option * bounds
  | IDef        of loc * decorator * async * defname * generics * params * ty option * bounds * block
  | ITask       of loc * decorator * defname * generics * params * sinks * bounds * block
  | IVal        of loc * decorator * name * ty option * expr
  | IExternType of loc * decorator * name * generics * bounds
  | IClass      of loc * decorator * name * generics * bounds * method_decls
  | IInstance   of loc * decorator * generics * path * tys * bounds * method_defs
  | IMod        of loc * decorator * name * items
  | IType       of loc * decorator * name * generics * ty * bounds
  | IUse        of loc * decorator * path * use_suffix option

and bounds = bound list
and bound = path * tys

and use_suffix =
  | UAlias of name
  | UGlob

and method_decls = method_decl list
and method_decl = name * generics * params * ty option * bounds

and method_defs = method_def list
and method_def = name * generics * params * ty option * bounds * block

and patterns = pattern list
and pattern =
  | PIgnore of loc
  | POr     of loc * pattern * pattern
  | PRecord of loc * pattern record
  | PTuple  of loc * patterns
  | PArray  of loc * patterns * pattern option
  | PConst  of loc * lit
  | PVar    of loc * name
  | PUnwrap of loc * name * pattern

and tys = ty list
and ty =
  | TFunc   of loc * tys * ty
  | TTuple  of loc * tys
  | TEnum   of loc * ty enum
  | TRecord of loc * ty record
  | TPath   of loc * path * tys
  | TArray  of loc * ty

and binop =
  | BAdd
  | BAnd
  | BBand
  | BBor
  | BBxor
  | BDiv
  | BEq
  | BGeq
  | BGt
  | BLeq
  | BLt
  | BMod
  | BMul
  | BMut
  | BNeq
  | BOr
  | BPow
  | BSub
  | BXor
  | BIn
  | BRExc
  | BRInc
  | BBy
  | BNotIn

and unop =
  | UNeg
  | UNot

and int_suffix = string
and float_suffix = string
and lit =
  | LInt    of loc * int * int_suffix option
  | LFloat  of loc * float * float_suffix option
  | LBool   of loc * bool
  | LString of loc * string
  | LUnit   of loc
  | LChar   of loc * char

and stmts = stmt list
and stmt =
  | SNoop of loc
  | SVal  of loc * param * expr
  | SVar  of loc * (name * ty option) * expr
  | SExpr of loc * expr

and exprs = expr list
and expr =
  | EAccess   of loc * expr * name
  | ECall     of loc * expr * exprs
  | ECast     of loc * expr * ty
  | EIf       of loc * expr * block * block option
  | ELit      of loc * lit
  | ELoop     of loc * block
  | ERecord   of loc * expr record
  | EEnwrap   of loc * name * expr
  | EReturn   of loc * expr option
  | EBreak    of loc * expr option
  | EContinue of loc
  (* NB: These expressions are desugared *)
  | EBinOpRef of loc * binop
  | EUnOp     of loc * unop * tys * expr
  | EArray    of loc * exprs * expr option
  | EBinOp    of loc * binop * tys * expr * expr
  | EBlock    of loc * block
  | EFor      of loc * pattern * expr * block
  | EFunc     of loc * params * block
  | EIfVal    of loc * pattern * expr * block * block option
  | EInvoke   of loc * expr * name * exprs
  | EMatch    of loc * expr * arms
  | EOn       of loc * receivers
  | EPath     of loc * path * tys
  | EProject  of loc * expr * index
  | ESelect   of loc * expr * expr
  | ETask     of loc * params * sinks * block
  | EThrow    of loc * expr
  | ETry      of loc * block * arms * block option
  | ETuple    of loc * exprs
  | EFrom     of loc * scans * steps
  | EAnon     of loc
  | EWhile    of loc * expr * block
  | EWhileVal of loc * pattern * expr * block

and receivers = receiver list
and receiver = pattern * expr * expr

and scans = scan list
and scan = pattern * scankind * expr
and scankind =
  | ScIn of loc
  | ScEq of loc

and steps = step list
and step =
  | SWhere of loc * expr
  | SJoin of loc * scan * join_on option
  | SGroup of loc * exprs * window option * reduces
  | SOrder of loc * (expr * ord) list
  | SYield of loc * expr

and join_on = expr

and window = window_step option * window_duration
and window_step = expr
and window_duration = expr

and reduces = reduce list
and reduce = expr * expr option (* Aggregation and Column *)

and ord =
  | OAsc
  | ODesc

let rec unop_name op =
  match op with
  | UNeg -> "neg"
  | UNot -> "not"

and binop_name op =
  match op with
  | BAdd -> "add"
  | BAnd -> "and"
  | BBand -> "band"
  | BBor -> "bor"
  | BBxor -> "bxor"
  | BDiv -> "div"
  | BGeq -> "geq"
  | BGt -> "gt"
  | BLeq -> "leq"
  | BLt -> "lt"
  | BMod -> "mod"
  | BMul -> "mul"
  | BNeq -> "neq"
  | BOr -> "or"
  | BPow -> "pow"
  | BSub -> "sub"
  | BXor -> "xor"
  | BIn -> "contains"
  | BNotIn -> "not_contains"
  | BRExc -> "rexc"
  | BRInc -> "rinc"
  | BEq -> "eq"
  | BMut -> "mut"
  | BBy -> "by"

and def_name d =
  match d with
  | DName x -> x
  | DBinOp (op, _) -> binop_name op
  | DUnOp (op, _) -> unop_name op

and item_loc p =
  match p with
  | IExternDef  (i, _, _, _, _, _, _, _) -> i
  | IDef        (i, _, _, _, _, _, _, _, _) -> i
  | ITask       (i, _, _, _, _, _, _, _) -> i
  | IVal        (i, _, _, _, _) -> i
  | IExternType (i, _, _, _, _) -> i
  | IClass      (i, _, _, _, _, _) -> i
  | IInstance   (i, _, _, _, _, _, _) -> i
  | IMod        (i, _, _, _) -> i
  | IType       (i, _, _, _, _, _) -> i
  | IUse        (i, _, _, _) -> i

and pat_loc p =
  match p with
  | PIgnore (i) -> i
  | POr (i, _, _) -> i
  | PRecord (i, _) -> i
  | PTuple (i, _) -> i
  | PArray (i, _, _) -> i
  | PConst (i, _) -> i
  | PVar (i, _) -> i
  | PUnwrap (i, _, _) -> i

and ty_loc t =
  match t with
  | TFunc (i, _, _) -> i
  | TTuple (i, _) -> i
  | TEnum (i, _) -> i
  | TRecord (i, _) -> i
  | TPath (i, _, _) -> i
  | TArray (i, _) -> i

and expr_loc e =
  match e with
  | EAccess (i, _, _) -> i
  | ECall (i, _, _) -> i
  | ECast (i, _, _) -> i
  | EIf (i, _, _, _) -> i
  | ELit (i, _) -> i
  | ELoop (i, _) -> i
  | ERecord (i, _) -> i
  | EEnwrap (i, _, _) -> i
  | EReturn (i, _) -> i
  | EBreak (i, _) -> i
  | EContinue (i) -> i
  | EBinOpRef (i, _) -> i
  | EUnOp (i, _, _, _) -> i
  | EArray (i, _, _) -> i
  | EBinOp (i, _, _, _, _) -> i
  | EBlock (i, _) -> i
  | EFor (i, _, _, _) -> i
  | EFunc (i, _, _) -> i
  | EIfVal (i, _, _, _, _) -> i
  | EInvoke (i, _, _, _) -> i
  | EMatch (i, _, _) -> i
  | EOn (i, _) -> i
  | EPath (i, _, _) -> i
  | EProject (i, _, _) -> i
  | ESelect (i, _, _) -> i
  | ETask (i, _, _, _) -> i
  | ETry (i, _, _, _) -> i
  | EThrow (i, _) -> i
  | ETuple (i, _) -> i
  | EFrom (i, _, _) -> i
  | EAnon (i) -> i
  | EWhile (i, _, _) -> i
  | EWhileVal (i, _, _, _) -> i

and name_of_item i =
  match i with
  | IExternDef  (_, _, _, d, _, _, _, _) -> Some (def_name d)
  | IDef        (_, _, _, d, _, _, _, _, _) -> Some (def_name d)
  | ITask       (_, _, d, _, _, _, _, _) -> Some (def_name d)
  | IVal        (_, _, x, _, _) -> Some x
  | IExternType (_, _, x, _, _) -> Some x
  | IClass      (_, _, x, _, _, _) -> Some x
  | IInstance   (_, _, _, _, _, _, _) -> None
  | IMod        (_, _, x, _) -> Some x
  | IType       (_, _, x, _, _, _) -> Some x
  | IUse        (_, _, _, _) -> None

(* Extracts a list of uses for all items in this item *)
and extract_prelude (i:item) =
  let rec items_of_mod xs acc i =
    match i with
    | IMod (_, _, x, is) ->
        is |> foldl (items_of_mod (x::xs)) acc
    | _ ->
        match name_of_item i with
        | Some x -> (x::xs)::acc
        | None -> acc
  in
  items_of_mod [] [] i |> map (fun xs -> IUse (NoLocStd, [], PAbs (List.rev xs), None))

(* Add items as a prelude to item *)
and add_prelude is0 i =
  match i with
  | IMod (loc, d, x, is1) ->
      let is1' = is1 |> map (add_prelude is0) in
      IMod (loc, d, x, is0 @ is1')
  | _ -> i

and loc_to_std l =
  match l with
  | Loc l -> LocStd l
  | NoLoc -> NoLocStd
  | NoLocStd -> l
  | LocStd _ -> l

and std_loc i =
  match i with
  | IExternDef (l, d, async, x, gs, ts, t, bs) -> IExternDef (loc_to_std l, d, async, x, gs, ts, t, bs)
  | IDef (l, d, async, x, gs, ps, t, bs, b) -> IDef (loc_to_std l, d, async, x, gs, ps, t, bs, b)
  | ITask (l, d, x, gs, ps, xts, bs, b) -> ITask (loc_to_std l, d, x, gs, ps, xts, bs, b)
  | IVal (l, d, x, t, e) -> IVal (loc_to_std l, d, x, t, e)
  | IExternType (l, d, x, bs, gs) -> IExternType (loc_to_std l, d, x, bs, gs)
  | IClass (l, d, x, gs, bs, method_decls) -> IClass (loc_to_std l, d, x, gs, bs, method_decls)
  | IInstance (l, d, gs, xs, ts, bs, method_defs) -> IInstance (loc_to_std l, d, gs, xs, ts, bs, method_defs)
  | IMod (l, d, x, is) -> IMod (loc_to_std l, d, x, is |> map std_loc)
  | IType (l, d, x, gs, bs, t) -> IType (loc_to_std l, d, x, gs, bs, t)
  | IUse (l, d, xs, use_suffix) -> IUse (loc_to_std l, d, xs, use_suffix)
