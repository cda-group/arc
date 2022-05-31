type token =
  | ParenL
  | ParenR
  | BrackL
  | BrackR
  | PoundBraceL
  | PoundParenL
  | BraceL
  | BraceR
  | AngleL
  | AngleR
(*= Operators ==============================================================*)
  | Neq
  | Percent
  | Star
  | StarStar
  | Plus
  | Comma
  | Minus
  | Dot
  | DotDot
  | DotDotEq
  | Slash
  | Colon
  | ColonColon
  | Semi
  | Leq
  | Eq
  | EqEq
  | Imply
  | Geq
  | AtSign
  | Underscore
  | Bar
(*= Keywords ================================================================*)
  | And
  | As
  | Async
  | Break
  | Band
  | Bor
  | Bxor
  | Catch
  | Class
  | Continue
  | Def
  | Desc
  | Duration
  | Else
  | Enum
  | Extern
  | Finally
  | For
  | From
  | Fun
  | Group
  | If
  | In
  | Instance
  | Join
  | Loop
  | Match
  | Mod
  | Neg
  | Not
  | On
  | Or
  | Of
  | Order
  | Return
  | Compute
  | Step
  | Task
  | Throw
  | Try
  | Type
  | Val
  | Var
  | Where
  | Window
  | While
  | Use
  | Xor
  | Yield
(*= Identifiers and Literals ================================================*)
  | Name of string
  | Int of int
  | IntSuffix of (int * Ast.int_suffix)
  | Float of float
  | FloatSuffix of (float * Ast.float_suffix)
  | Bool of bool
  | Char of char
  | String of string
  | Unit
(*   | DurationNs of int *)
(*   | DurationUs of int *)
(*   | DurationMs of int *)
(*   | DurationS of int *)
(*   | DurationM of int *)
(*   | DurationH of int *)
(*   | DurationD of int *)
(*   | DurationW of int *)
(*     LitDurationMo, *)
(*     LitDurationY, *)
(*   | Date of string *)
(*   | DateTime of string *)
(*   | DateTimeZone of string *)
  | Eof
