type ast = items
and name = string
and path = name list
and arms = arm list
and arm = pattern * expr
and params = param list
and param = pattern * ty option
and index = int
and 't fields = 't field list
and 't field = name * 't option
and 't record = 't fields * 't option
and port = name * ty
and variants = variant list
and variant = name * tys
and block = stmts * expr option
and generics = generic list
and generic = name

and defname =
  | DName of name
  | DUnOp of unop
  | DBinOp of binop

and decorator = lit fields

and items = item list
and item =
  | IExternDef  of decorator * defname * generics * tys * ty option
  | IDef        of decorator * defname * generics * params * ty option * block option
  | ITask       of decorator * defname * generics * params * params * block option
  | IVal        of decorator * name * ty option * expr
  | IEnum       of decorator * name * generics * variants
  | IExternType of decorator * name * generics
  | IClass      of decorator * name * generics * decls
  | IInstance   of decorator * generics * path * tys * defs
  | IMod        of decorator * name * items
  | ITypeAlias  of decorator * name * generics * ty
  | IUse        of decorator * path * name option

and decls = decl list
and decl = name * generics * params * ty option

and defs = def list
and def = name * generics * params * ty option * block

and patterns = pattern list
and pattern =
  | PIgnore
  | POr      of pattern * pattern
  | PRecord  of pattern record
  | PTuple   of patterns
  | PConst   of lit
  | PVar     of name
  | PUnwrap  of path * patterns

and tys = ty list
and ty =
  | TFunc   of tys * ty
  | TTuple  of tys
  | TRecord of ty record
  | TPath   of path * tys
  | TArray  of ty

and binop =
  | BAdd | BAddf
  | BAnd
  | BBand
  | BBor
  | BBxor
  | BDiv | BDivf
  | BEq  | BEqf
  | BGeq | BGeqf
  | BGt  | BGtf
  | BLeq | BLeqf
  | BLt  | BLtf
  | BMod | BModf
  | BMul | BMulf
  | BMut
  | BNeq | BNeqf
  | BOr
  | BPow | BPowf
  | BSub | BSubf
  | BXor
  | BIn
  | BRExc
  | BRInc
  | BBy
  | BNotIn

and unop =
  | UNeg | UNegf
  | UNot

and size = int
and sign = bool
and lit =
  | LInt of int * (sign * size) option
  | LFloat of float * size option
  | LBool of bool
  | LString of string
  | LUnit
  | LChar of char

and stmts = stmt list
and stmt =
  | SNoop
  | SVal  of param * expr
  | SVar  of (name * ty option) * expr
  | SExpr of expr

and exprs = expr list
and expr =
  | EAccess   of expr * name
  | ECall     of expr * exprs
  | ECast     of expr * ty
  | EIf       of expr * block * block option
  | ELit      of lit
  | ELoop     of block
  | ERecord   of expr record
  | EReturn   of expr option
  | EBreak    of expr option
  | EContinue
  (* NB: These expressions are desugared *)
  | EBinOpRef of binop
  | EUnOp     of unop * expr
  | EArray    of exprs * expr option
  | EBinOp    of binop * expr * expr
  | EBlock    of block
  | ECompr    of expr * (pattern * expr) * clauses
  | EFor      of pattern * expr * block
  | EFunc     of params * block
  | EIfVal    of pattern * expr * block * block option
  | EInvoke   of expr * name * exprs
  | EMatch    of expr * arms
  | EReceive  of expr
  | EEmit     of expr * expr
  | EOn       of receivers
  | EPath     of path * tys
  | EProject  of expr * index
  | ESelect   of expr * expr
  | ETask     of params * block
  | ETuple    of exprs
  | EFrom     of scans * steps
  | EAnon
  | EWhile    of expr * block
  | EWhileVal of pattern * expr * block

and receivers = receiver list
and receiver = pattern * expr * expr

and clauses = clause list
and clause =
  | CFor of pattern * expr
  | CIf of expr

and scans = scan list
and scan = pattern * scankind * expr
and scankind =
  | ScIn
  | ScEq

and steps = step list
and step =
  | SWhere of expr
  | SJoin of scan * expr option
  | SGroup of exprs * window option * reduces
  | SOrder of (expr * ord) list
  | SYield of expr

and window = expr option * expr (* Step and Duration *)

and reduces = reduce list
and reduce = expr * expr option (* Aggregation and Column *)

and ord =
  | OAsc
  | ODesc

let rec unop_name op =
  match op with
  | UNeg -> "neg" | UNegf -> "negf"
  | UNot -> "not"

and binop_name op =
  match op with
  | BAdd -> "add" | BAddf -> "addf"
  | BAnd -> "and"
  | BBand -> "band"
  | BBor -> "bor"
  | BBxor -> "bxor"
  | BDiv -> "div" | BDivf -> "divf"
  | BGeq -> "geq" | BGeqf -> "geqf"
  | BGt -> "gt"   | BGtf -> "gtf"
  | BLeq -> "leq" | BLeqf -> "leqf"
  | BLt -> "lt"   | BLtf -> "ltf"
  | BMod -> "mod" | BModf -> "modf"
  | BMul -> "mul" | BMulf -> "mulf"
  | BNeq -> "neq" | BNeqf -> "neqf"
  | BOr -> "or"
  | BPow -> "pow" | BPowf -> "powf"
  | BSub -> "sub" | BSubf -> "subf"
  | BXor -> "xor"
  | BIn -> "contains"
  | BNotIn -> "not_contains"
  | BRExc -> "rexc"
  | BRInc -> "rinc"
  | BEq -> "eq" | BEqf -> "eq"
  | BMut -> "mut"
  | BBy -> "by"

and def_name d =
  match d with
  | DName x -> x
  | DBinOp op -> binop_name op
  | DUnOp op -> unop_name op
