open Error

type mlir = (symbol * item) list
and names = name list
and name = string
and value = name
and symbol = name
and 't fields = 't field list
and 't field = name * 't
and 't variants = 't variant list
and 't variant = name * 't
and ssas = ssa list
and ssa = param option * op
and args = arg list
and arg = value * ty
and params = param list
and param = value * ty
and block = ssas
and async = bool
and item =
  | IExternFunc of loc * symbol * async * params * ty
  | IFunc       of loc * params * ty option * block
and tys = ty list
and ty =
  | TFunc    of tys * ty
  | TRecord  of ty fields
  | TEnum    of ty fields
  | TAdt     of name * tys
  | TNative  of name
and ops = op list
and op =
  | OAccess   of loc * arg * name
  | OUpdate   of loc * arg * name * arg
  | OCallExpr of loc * arg * args
  | OCallItem of loc * symbol * args
  | OEnwrap   of loc * name * arg option
  | OIf       of loc * arg * block * block
  | OCheck    of loc * name * arg
  | OConst    of loc * const
  | OLoop     of loc * block
  | ORecord   of loc * names * tys
  | OUnwrap   of loc * name * arg
  | OReturn   of loc * arg option
  | OBreak    of loc * arg option
  | OContinue of loc
  | OYield    of loc
  | OResult   of loc * arg option
  | OSpawn    of loc * symbol * args
and const =
  | CBool  of bool
  | CFun   of symbol
  | CInt   of int
  | CFloat of float
  | CAdt   of string

let item_loc i =
  match i with
  | IExternFunc (i, _, _, _, _) -> i
  | IFunc       (i, _, _, _) -> i
