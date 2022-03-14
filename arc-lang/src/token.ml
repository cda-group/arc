type token =
  | ParenL
  | ParenR
  | BrackL
  | BrackR
  | PoundBraceL
  | BraceL
  | BraceR
  | AngleL
  | AngleR
(*= Operators ==============================================================*)
  | Bang
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
(*= Float extensions =======================================================*)
  | EqEqSuffix of string
  | GeqSuffix of string
  | GtSuffix of string
  | LeqSuffix of string
  | LtSuffix of string
  | MinusSuffix of string
  | NeqSuffix of string
  | PercentSuffix of string
  | PlusSuffix of string
  | SlashSuffix of string
  | StarSuffix of string
  | StarStarSuffix of string
(*= Keywords ================================================================*)
  | And
  | As
  | Break
  | Band
  | Bor
  | Bxor
  | Class
  | Continue
  | Def
  | Desc
  | Duration
  | Else
  | Enum
  | Extern
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
  | Not
  | On
  | Or
  | Of
  | Order
  | Return
  | Reduce
  | Step
  | Task
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
  | Float of float
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
