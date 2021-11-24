type ast = item list
and name = string
and path = name list
and arm = pattern * expr
and param = pattern * ty option
and basic_param = name * ty
and index = int
and 't field = name * 't
and port = name * ty
and variant = name * ty list
and block = stmt list * expr option
and generic = name
and interface =
  | PTagged of port list
  | PSingle of ty

and item =
  | IVal        of name * ty option * expr
  | IEnum       of name * generic list * variant list
  | IExternDef  of name * generic list * basic_param list * ty
  | IExternType of name * generic list
  | IClass      of name * generic list * decl list
  | IInstance   of generic list * path * ty list * def list
  | IDef        of name * generic list * param list * ty option * block option
  | ITask       of name * generic list * param list * interface * interface * block option
  | IMod        of name * item list
  | ITypeAlias  of name * ty
  | IUse        of path * name option

and decl = name * generic list * param list * ty option
and def = name * generic list * param list * ty option * block

and pattern =
  | PIgnore
  | POr      of pattern * pattern
  | PRecord  of pattern option field list
  | PTuple   of pattern list
  | PConst   of lit
  | PVar     of name
  | PUnwrap  of path * pattern list

and ty =
  | TFunc   of ty list * ty
  | TTuple  of ty list
  | TRecord of ty field list * ty option
  | TPath   of path * ty list
  | TArray  of ty

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
  (* NB: These ops are desugared *)
  | BBy
  | BNotIn
  | BPipe

and unop =
  | UNeg
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

and stmt =
  | SNoop
  | SVal  of param * expr
  | SVar  of (name * ty option) * expr
  | SExpr of expr

and expr =
  | EAccess   of expr * name
  | EAfter    of expr * block
  | EEvery    of expr * block
  | ECall     of expr * expr list
  | ECast     of expr * ty
  | EEmit     of expr
  | EIf       of expr * block * block option
  | ELit      of lit
  | ELoop     of block
  | ERecord   of expr option field list * expr option
  | EReturn   of expr option
  | EBreak    of expr option
  | EContinue
  (* NB: These expressions are desugared *)
  | EUnOp     of unop * expr
  | EArray    of expr list * expr option
  | EBinOp    of binop * expr * expr
  | EBlock    of block
  | ECompr    of expr * (pattern * expr) * clause list
  | EFor      of pattern * expr * block
  | EFunc     of param list * expr
  | EIfVal    of pattern * expr * block * block option
  | EInvoke   of expr * name * expr list
  | EMatch    of expr * arm list
  | EOn       of arm list
  | EPath     of path * ty list
  | EProject  of expr * index
  | ESelect   of expr * expr
  | ETask     of expr
  | ETuple    of expr list
  | EFrom     of scan list * step list

and clause =
  | CFor of pattern * expr
  | CIf of expr

and scan = pattern * scankind * expr
and scankind =
  | ScIn
  | ScEq

and step =
  | SWhere of expr
  | SJoin of scan * expr option
  | SGroup of expr list * (expr * expr option) list
  | SOrder of (expr * ord) list
  | SYield of expr

and ord =
  | OAsc
  | ODesc
