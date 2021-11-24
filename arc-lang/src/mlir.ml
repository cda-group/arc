type mlir = (path * item) list
and names = name list
and name = string
and path = name
and 't fields = 't field list
and 't field = name * 't
and ops = op list
and op = param option * expr
and args = arg list
and arg = name * ty
and params = param list
and param = name * ty
and block = ops
and item =
  | IAssign     of ty * block
  | IExternFunc of params * ty
  | IFunc       of params * ty option * block
  | ITask       of params * params * block
and tys = ty list
and ty =
  | TFunc    of tys * ty
  | TRecord  of ty fields
  | TEnum    of ty fields
  | TAdt     of path * tys
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
  | ECall     of arg * args
  | EEmit     of arg * arg
  | EEnwrap   of path * arg option
  | EIf       of arg * block * block
  | EIs       of path * arg
  | EConst    of const
  | ELoop     of block
  | ERecord   of names * tys
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
  | CAdt   of string
