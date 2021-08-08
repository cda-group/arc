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
  | IExternFunc of name * generic list * basic_param list * ty
  | IExternType of name * generic list
  | IFunc       of name * generic list * param list * ty option * block option
  | ITask       of name * generic list * param list * interface * interface * block option
  | IMod        of name * item list
  | ITypeAlias  of name * ty
  | IUse        of path * name option

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
  | TStream of ty

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
  | EArray    of expr list
  | EBinOp    of binop * expr * expr
  | ECall     of expr * expr list
  | EInvoke   of expr * name * expr list
  | ECast     of expr * ty
  | EEmit     of expr
  | EIf       of expr * block * block option
  | ELit      of lit
  | ELog      of expr
  | ELoop     of block
  | EOn       of arm list
  | ESelect   of expr * expr
  | ERecord   of expr option field list
  | EUnOp     of unop * expr
  | EReturn   of expr option
  | EBreak    of expr option
  | EContinue
  (* NB: These expressions are desugared *)
  | ETuple    of expr list
  | EProject  of expr * index
  | EBlock    of block
  | EFunc     of param list * expr
  | ETask     of expr
  | EIfVal    of pattern * expr * block * block option
  | EFor      of pattern * expr * block
  | EMatch    of expr * arm list
  | ECompr    of expr * (pattern * expr) * clause list
  | EPath     of path * ty list
  | EWith     of expr * expr option field list

and clause =
  | CFor of pattern * expr
  | CIf of expr

and mut =
  | MVal (* Immutable *)
  | MVar (* Mutable *)

