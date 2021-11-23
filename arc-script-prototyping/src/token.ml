type token =
  | ParenL
  | ParenR
  | ParenLR
  | BrackL
  | BrackR
  | BrackLR
  | PercentBraceL
  | BraceL
  | BraceR
  | BraceLR
  | AngleL
  | AngleR
  | AngleLR
(*= Operators ==============================================================*)
  | Neq
  | Percent
  | Star
  | StarStar
  | Plus
  | Comma
  | Minus
  | ArrowR
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
  | BarBar
  | Ampersand
(*= Retired Operators =======================================================*)
(*     Tilde, *)
(*     Pipe, *)
(*     Bang, *)
(*     Dollar, *)
(*     AmpAmp, *)
(*     SemiSemi, *)
(*     ArrowL, *)
(*     Qm, *)
(*     QmQmQm, *)
(*     Caret, *)
(*= Keywords ================================================================*)
  | After
  | And
  | As
  | Break
  | Band
  | Bor
  | Bxor
  | By
  | Class
  | Continue
  | Def
  | Else
  | Emit
  | Enum
  | Every
  | Extern
  | For
  | From
  | Fun
  | If
  | In
  | Instance
  | Is
  | Let
  | Loop
  | Match
  | Mod
  | Not
  | On
  | Or
  | Return
  | Task
  | Type
  | Val
(*   | With *)
  | Var
  | Unwrap
  | Enwrap
  | Use
  | Xor
(*= Reserved Keywords =======================================================*)
(*     Add, *)
(*     Box, *)
(*     Do, *)
(*     End, *)
(*     Of, *)
(*     Port, *)
(*     Pub, *)
(*     Reduce, *)
(*     Shutdown, *)
(*     Sink, *)
(*     Source, *)
(*     Startup, *)
(*     State, *)
(*     Then, *)
(*     Timeout, *)
(*     Timer, *)
(*     Trigger, *)
(*     Where, *)
(*= Identifiers and Literals ================================================*)
  | Name of string
  | Int of int
  | Float of float
  | Bool of bool
  | Char of char
  | String of string
  | Unit
  | DurationNs of int
  | DurationUs of int
  | DurationMs of int
  | DurationS of int
  | DurationM of int
  | DurationH of int
  | DurationD of int
  | DurationW of int
(*     LitDurationMo, *)
(*     LitDurationY, *)
  | Date of string
  | DateTime of string
  | DateTimeZone of string
  | Eof
