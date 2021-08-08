type mlir = (path * item) list
and name = string
and path = name
and 't field = name * 't
and op = param option * expr
and arg = name * ty
and param = name * ty
and block = op list
and item =
  | IAssign     of ty * block
  | IExternFunc of param list * ty
  | IFunc       of param list * ty * block
  | ITask       of param list * block
and ty =
  | TFunc    of ty list * ty
  | TRecord  of ty field list
  | TEnum    of ty field list
  | TAdt     of path * ty list
  | TStream  of ty
  | TNative  of name
and num =
  | NFlt
  | NInt
and eq =
  | EqFlt
  | EqInt
  | EqBool
and binop =
  | BAdd  of num
  | BAnd
  | BBand
  | BBor
  | BBxor
  | BDiv  of num
  | BEqu  of eq
  | BGeq  of num
  | BGt   of num
  | BLeq  of num
  | BLt   of num
  | BMod  of num
  | BMul  of num
  | BMut
  | BNeq  of eq
  | BOr
  | BPow  of num
  | BSub  of num
  | BXor
and expr =
  | EAccess   of arg * name
  | EBinOp    of binop * arg * arg
  | ECall     of arg * arg list
  | EEmit     of arg * arg
  | EEnwrap   of path * arg option
  | EIf       of arg * block * block
  | EIs       of path * arg
  | EConst    of const
  | ELoop     of block
  | ERecord   of arg field list
  | EReceive  of arg
  | EUnwrap   of path * arg
  | EReturn   of arg option
  | EBreak    of arg option
  | EContinue
  | EYield
  | EResult   of arg option
  | ENoop
and const =
  | CBool  of bool
  | CFun   of path
  | CInt   of int
  | CFloat of float
